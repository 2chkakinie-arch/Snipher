"""未知文字の学習。

LFM2.5-1.2B-JP の学習済みパラメータをそのまま活用して、トークナイザが
保持できない文字（希少漢字・絵文字・新語など）を会話に参加させる。

レベル1（即時学習, gradient 不要・数百ミリ秒）:
    予約スロットに新トークンを割り当て、その埋め込み行を
    「トークナイザがその文字を分解した既知断片の埋め込み（事前学習済み
    パラメータ）の平均」で初期化する。モデルはその文字を UNK ではなく
    1 トークンとして読めるようになる。

レベル2（深学習, 埋め込みのみの少数ステップ勾配更新）:
    例文を与えて、新トークンの行だけを causal LM の目的で数ステップ更新
    する。本体の重みは完全に凍結するので、破壊的でなく高速（CPU でも
    数十秒）。更新後の行はディスクに永続化され、再起動後も有効。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# 例文が無いときに使う自動テンプレート（日常会話の型）
AUTO_EXAMPLE_TEMPLATES = [
    "これは{x}です。",
    "私は{x}が好きです。",
    "最近{x}に興味がある。",
    "{x}について話しましょう。",
]


def _head_weight_tensor(model):
    """lm_head の生の重みテンソルを返す。(tensor, is_plain) — 量子化済みなら (None, False)。"""
    head = getattr(model, "lm_head", None)
    if head is None:
        return None, False
    w = getattr(head, "weight", None)
    if w is None or callable(w):  # 動的量子化モジュールでは weight はメソッド
        return None, False
    return w, True


# ---------------------------------------------------------------------- #
# 永続化
# ---------------------------------------------------------------------- #
@dataclass
class LearnedVocabStore:
    """学習済み文字と埋め込み行の永続化（JSON + safetensors）。"""

    store_dir: Path
    chars: dict = field(default_factory=dict)     # char -> meta dict
    rows: dict = field(default_factory=dict)      # token_id(str) -> [float, ...]
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def json_path(self) -> Path:
        return self.store_dir / "learned_vocab.json"

    @property
    def rows_path(self) -> Path:
        return self.store_dir / "learned_rows.safetensors"

    def load(self) -> None:
        self.chars, self.rows = {}, {}
        if self.json_path.exists():
            try:
                self.chars = json.loads(self.json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("learned_vocab.json 読込失敗: %s", exc)
        if self.rows_path.exists():
            try:
                from safetensors.torch import load_file
                import torch

                blob = load_file(str(self.rows_path))
                ids = blob["ids"].tolist()
                mat = blob["rows"].tolist()
                self.rows = {str(i): r for i, r in zip(ids, mat)}
            except Exception as exc:
                log.warning("learned_rows.safetensors 読込失敗: %s", exc)
        # rows の無いエントリは破棄
        self.chars = {c: m for c, m in self.chars.items() if str(m.get("token_id")) in self.rows}

    def save(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.json_path.write_text(
                json.dumps(self.chars, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            try:
                import torch
                from safetensors.torch import save_file

                if self.rows:
                    ids = sorted(int(k) for k in self.rows)
                    mat = torch.tensor([self.rows[str(i)] for i in ids], dtype=torch.float32)
                    save_file(
                        {"ids": torch.tensor(ids, dtype=torch.int64), "rows": mat},
                        str(self.rows_path),
                    )
            except Exception as exc:
                log.error("埋め込み行の保存に失敗: %s", exc)

    # ------------------------------------------------------------------ #
    def token_ids(self) -> list[int]:
        return sorted(int(m["token_id"]) for m in self.chars.values())

    def char_of(self, token_id: int) -> str | None:
        for c, m in self.chars.items():
            if int(m.get("token_id", -1)) == token_id:
                return c
        return None

    def used_ids(self) -> set[int]:
        return {int(m["token_id"]) for m in self.chars.values()}

    def add(self, char: str, token_id: int, row: list[float], meta: dict) -> None:
        self.chars[char] = {"token_id": token_id, "created": time.time(), **meta}
        self.rows[str(token_id)] = row

    def piece_ids_of_char(self, meta: dict) -> list[int]:
        """学習時に保存したトークナイザ断片 id 列（無ければ空）。"""
        return [int(x) for x in meta.get("piece_ids") or []]

    def remove_char(self, char: str) -> int | None:
        meta = self.chars.pop(char, None)
        if meta is None:
            return None
        tid = int(meta["token_id"])
        self.rows.pop(str(tid), None)
        return tid

    def clear(self) -> None:
        self.chars, self.rows = {}, {}
        for p in (self.json_path, self.rows_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------- #
# 学習器
# ---------------------------------------------------------------------- #
class UnknownCharLearner:
    """予約トークン枠への文字割当と埋め込み学習。"""

    def __init__(self, store: LearnedVocabStore, reserved: int = 256):
        self.store = store
        self.reserved = int(reserved)
        self.base_vocab: int | None = None  # engine が load 時に設定

    # ------------------------------------------------------------------ #
    # 割当
    # ------------------------------------------------------------------ #
    def next_free_id(self) -> int | None:
        assert self.base_vocab is not None
        used = self.store.used_ids()
        for off in range(self.reserved):
            tid = self.base_vocab + off
            if tid not in used:
                return tid
        return None

    # ------------------------------------------------------------------ #
    # レベル1: 即時学習（事前学習済み断片埋め込みの平均で初期化）
    # ------------------------------------------------------------------ #
    def compose_row(self, tokenizer, embedding_weight, char: str, pieces: list[str] | None = None):
        """char の初期行 = 既知断片トークンの埋め込み平均（事前学習済みパラメータの合成）。"""
        import torch

        with torch.no_grad():
            if pieces:
                ids: list[int] = []
                for p in pieces:
                    got = tokenizer.encode(p, add_special_tokens=False)
                    ids.extend(int(i) for i in got if 0 <= int(i) < self.base_vocab)
            else:
                try:
                    got = tokenizer.encode(char, add_special_tokens=False)
                except TypeError:
                    got = tokenizer.encode(char)
                ids = [int(i) for i in got if 0 <= int(i) < self.base_vocab]
            if ids:
                rows = embedding_weight[torch.tensor(ids, dtype=torch.long)].to(torch.float32)
                return rows.mean(dim=0)
            # フォールバック: 平仮名の平均（中立な「日本語っぽさ」）
            hira = [ord(c) for c in "あいうえおかきくけこさしすせそ"]
            rows = embedding_weight[torch.tensor(hira, dtype=torch.long)].to(torch.float32)
            return rows.mean(dim=0)

    def learn_instant(
        self,
        tokenizer,
        model,
        char: str,
        pieces: list[str] | None = None,
        source: str = "manual",
    ) -> dict:
        """モデルを直接書き換えて即座に使えるようにする（量子化モデルでも可）。"""
        import torch

        if self.base_vocab is None:
            raise RuntimeError("learner が model に未接続です")
        if char in self.store.chars:
            tid = int(self.store.chars[char]["token_id"])
            return {"char": char, "token_id": tid, "already": True}

        tid = self.next_free_id()
        if tid is None:
            raise RuntimeError("予約トークン枠が一杯です（SNIPHER_LFM_RESERVED を増やしてください）")

        emb = model.get_input_embeddings().weight
        row = self.compose_row(tokenizer, emb, char, pieces)
        with torch.no_grad():
            emb[tid] = row.to(emb.dtype)
            # untied の場合は出力側にも書き込む（量子化済みならスキップ）
            head_w, is_plain = _head_weight_tensor(model)
            if head_w is not None and head_w.data_ptr() != emb.data_ptr():
                if tid < head_w.shape[0]:
                    head_w[tid] = row.to(head_w.dtype)

        self.store.add(
            char,
            tid,
            [float(x) for x in row.tolist()],
            {"source": source, "pieces": pieces or [], "piece_ids": self.piece_ids(tokenizer, char)},
        )
        self.store.save()
        return {"char": char, "token_id": tid, "already": False, "initialized_from": len(pieces or [])}

    # ------------------------------------------------------------------ #
    # テキスト → 予約トークン写像
    # ------------------------------------------------------------------ #
    def piece_ids(self, tokenizer, char: str) -> list[int]:
        try:
            return [int(i) for i in tokenizer.encode(char, add_special_tokens=False)]
        except TypeError:
            return [int(i) for i in tokenizer.encode(char)]

    def map_ids(self, tokenizer, ids: list[int]) -> list[int]:
        """トークン列中の「未知文字の断片列」を学習済み予約トークンに置換する。

        これによりモデルは 𠮷 や 🚀 を 1 つの意味トークンとして読める。
        """
        if not self.store.chars or not ids:
            return ids
        seqs: list[tuple[tuple[int, ...], int]] = []
        for ch, meta in self.store.chars.items():
            pids = self.store.piece_ids_of_char(meta) or self.piece_ids(tokenizer, ch)
            if pids:
                seqs.append((tuple(pids), int(meta["token_id"])))
        if not seqs:
            return ids
        out = list(ids)
        for pids, tid in sorted(seqs, key=lambda x: -len(x[0])):
            n = len(pids)
            if n == 0 or n > len(out):
                continue
            res: list[int] = []
            i = 0
            while i < len(out):
                if tuple(out[i : i + n]) == pids:
                    res.append(tid)
                    i += n
                else:
                    res.append(out[i])
                    i += 1
            out = res
        return out

    # ------------------------------------------------------------------ #
    # レベル2: 深学習（埋め込み行のみを少数ステップの勾配で更新）
    # ------------------------------------------------------------------ #
    def build_examples(self, tokenizer, chars: list[str], examples: list[str], per_char: int = 4):
        """学習用の文リストを作る。"""
        texts = [e for e in (x.strip() for x in examples) if e]
        for ch in chars:
            made = 0
            for tpl in AUTO_EXAMPLE_TEMPLATES:
                s = tpl.format(x=ch)
                if s not in texts:
                    texts.append(s)
                    made += 1
                if made >= per_char:
                    break
        return texts[:64]

    def train_deep(
        self,
        tokenizer,
        model,
        chars: list[str],
        examples: list[str] | None = None,
        steps: int = 12,
        lr: float = 3e-3,
        on_progress=None,
    ) -> dict:
        """新トークンの行のみ AdamW で数ステップ更新する。本体は完全凍結。"""
        import torch
        import torch.nn as nn

        if self.base_vocab is None:
            raise RuntimeError("learner が model に未接続です")

        # 未学習文字は先にレベル1で初期化
        info_init = []
        for ch in chars:
            if ch not in self.store.chars:
                rep_pieces = None
                try:
                    from .vocab import classify_char

                    rep = classify_char(tokenizer, ch)
                    rep_pieces = rep.pieces if (rep and rep.is_unknown) else None
                except Exception:
                    rep_pieces = None
                self.learn_instant(tokenizer, model, ch, pieces=rep_pieces, source="deep")
            info_init.append(ch)

        learned_ids = {ch: int(self.store.chars[ch]["token_id"]) for ch in chars if ch in self.store.chars}
        if not learned_ids:
            raise RuntimeError("学習対象の文字がありません")

        texts = self.build_examples(tokenizer, list(learned_ids), examples or [])
        if not texts:
            raise RuntimeError("学習用の例文を作成できませんでした")

        device = next(model.parameters()).device
        n_new = self.reserved
        hidden = model.get_input_embeddings().weight.shape[1]
        init = torch.zeros(n_new, hidden, dtype=torch.float32)
        for tid, row in self.store.rows.items():
            t = int(tid)
            if self.base_vocab <= t < self.base_vocab + n_new:
                init[t - self.base_vocab] = torch.tensor(row, dtype=torch.float32)
        rows_param = nn.Parameter(init.clone())
        id2slot = {tid: i for i, tid in enumerate(sorted(learned_ids.values()))}
        slot_of_token = {}
        for tid in learned_ids.values():
            slot_of_token[tid] = id2slot[tid]

        # バッチ作成（未知文字は予約トークンへ写像してから使う）
        batch_ids: list[list[int]] = []
        for t in texts:
            enc = tokenizer(t, add_special_tokens=True, truncation=True, max_length=128)
            if enc["input_ids"]:
                batch_ids.append(self.map_ids(tokenizer, list(enc["input_ids"])))
        maxlen = max(len(x) for x in batch_ids)
        pad_id = tokenizer.pad_token_id or 0
        input_ids = torch.full((len(batch_ids), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch_ids), maxlen), dtype=torch.long)
        for i, seq in enumerate(batch_ids):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[i, : len(seq)] = 1

        # 学習位置マスク: 未知文字の近く（前2トークン以内）で、かつ
        # 予測先が既知語彙内の位置だけを損失に使う
        learned_set = torch.tensor(sorted(slot_of_token), dtype=torch.long)
        is_learned = torch.isin(input_ids, learned_set)
        near = torch.zeros_like(is_learned)
        for k in range(3):
            near |= torch.roll(is_learned, shifts=k, dims=1)
        targets = input_ids.roll(shifts=-1, dims=1)
        valid = near & (targets < self.base_vocab) & attn.bool()
        if valid.sum() == 0:  # まれに全て除外された場合
            valid = attn.bool() & (targets < self.base_vocab)

        emb_layer = model.get_input_embeddings()
        base_emb = emb_layer.weight
        opt = torch.optim.AdamW([rows_param], lr=lr)
        model.eval()

        losses = []
        t0 = time.time()
        for step in range(max(1, steps)):
            opt.zero_grad(set_to_none=True)
            emb_out = base_emb[input_ids]  # 凍結された既知埋め込み
            mask = input_ids >= self.base_vocab
            if mask.any():
                slot_idx = (input_ids - self.base_vocab).clamp(0, n_new - 1)
                delta = torch.zeros_like(emb_out)
                delta[mask] = rows_param[slot_idx[mask]].to(emb_out.dtype)
                emb_out = emb_out + delta
            out = model.model(
                inputs_embeds=emb_out,
                attention_mask=attn,
                use_cache=False,
            )
            logits = model.lm_head(out.last_hidden_state)
            logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            tgt = targets[:, :-1]
            vmask = valid[:, :-1]
            picked = logp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            loss = -(picked * vmask).sum() / vmask.sum().clamp_min(1)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if on_progress:
                on_progress(step + 1, steps, losses[-1])

        # 学習した行を store へ書き戻し（ 即時学習パスにも反映される）
        with torch.no_grad():
            for tid, slot in slot_of_token.items():
                self.store.rows[str(tid)] = [float(x) for x in rows_param[slot].tolist()]
        self.store.save()

        return {
            "chars": list(learned_ids),
            "examples": len(texts),
            "steps": len(losses),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "seconds": round(time.time() - t0, 2),
        }

    # ------------------------------------------------------------------ #
    # モデルへの適用（起動時に学習済み行を復元）
    # ------------------------------------------------------------------ #
    def apply_persisted(self, model) -> int:
        """保存済みの行をモデルの埋め込みに書き戻す。適用件数を返す。"""
        import torch

        if not self.store.rows or self.base_vocab is None:
            return 0
        emb = model.get_input_embeddings().weight
        head_w, is_plain = _head_weight_tensor(model)
        untied = head_w is not None and head_w.data_ptr() != emb.data_ptr()
        n = 0
        with torch.no_grad():
            for tid_s, row in self.store.rows.items():
                tid = int(tid_s)
                if not (self.base_vocab <= tid < self.base_vocab + self.reserved):
                    continue
                if tid >= emb.shape[0]:
                    continue
                emb[tid] = torch.tensor(row, dtype=emb.dtype)
                if untied and tid < head_w.shape[0]:
                    head_w[tid] = torch.tensor(row, dtype=head_w.dtype)
                n += 1
        return n
