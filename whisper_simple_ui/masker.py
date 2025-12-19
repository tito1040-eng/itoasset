#!/usr/bin/env python3
"""
Regex-based PII masker for Japanese text (no external NLP dependencies).

- Entities:
  ADDRESS, POSTAL, PHONE, EMAIL, PERSON (simple heuristics)
- Safe filters: Exclude times/dates/money/percent/units to reduce FP
- Tokens are customizable via arguments or env JSON MASK_TOKENS
"""

import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Pattern

@dataclass
class MaskTokens:
    DEFAULT: str = "<PII>"
    ADDRESS: str = "<ADDRESS>"
    POSTAL: str = "<POSTAL_CODE>"
    PHONE: str = "<PHONE>"
    EMAIL: str = "<EMAIL>"


def load_tokens() -> MaskTokens:
    env = os.getenv("MASK_TOKENS")
    if env:
        try:
            d = json.loads(env)
            return MaskTokens(
                DEFAULT=d.get("DEFAULT", "<PII>"),
                ADDRESS=d.get("ADDRESS", "<ADDRESS>"),
                POSTAL=d.get("POSTAL", "<POSTAL_CODE>"),
                PHONE=d.get("PHONE", "<PHONE>"),
                EMAIL=d.get("EMAIL", "<EMAIL>"),
            )
        except Exception:
            pass
    return MaskTokens()


# Patterns
HYPH = "-−ー―–—‑‐"
DIG = "0-9０-９"
PREFS = "|".join([
    "北海道","青森県","岩手県","宮城県","秋田県","山形県","福島県","茨城県","栃木県","群馬県","埼玉県","千葉県","東京都","神奈川県",
    "新潟県","富山県","石川県","福井県","山梨県","長野県","岐阜県","静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県",
    "奈良県","和歌山県","鳥取県","島根県","岡山県","広島県","山口県","徳島県","香川県","愛媛県","高知県","福岡県","佐賀県",
    "長崎県","熊本県","大分県","宮崎県","鹿児島県","沖縄県",
])

PAT_ADDRESS = re.compile(rf"(?:〒\s*[{DIG}]{{3}}[{HYPH}]?[{DIG}]{{4}}\s*)?(?:{PREFS})(?:[^\n\r]*?)(?:市|区|郡|町|村)(?:[^\n\r、。]{{0,80}})")
PAT_POSTAL = re.compile(rf"(?:〒\s*)?[{DIG}]{{3}}(?:[{HYPH}]?)[{DIG}]{{4}}")
PAT_PHONE = re.compile(rf"(?:(?:0[5789]0)[{HYPH}]?[{DIG}]{{3,4}}[{HYPH}]?[{DIG}]{{4}})|(?:0[1-9][0-9]{{0,3}}[{HYPH}][{DIG}]{{2,4}}[{HYPH}][{DIG}]{{3,4}})|(?:\+81\s?0?[{DIG}]{{1,4}}[{HYPH}][{DIG}]{{2,4}}[{HYPH}][{DIG}]{{3,4}})")
PAT_EMAIL = re.compile(r"[A-Za-z0-9._%+\-Ａ-Ｚａ-ｚ０-９＿％＋－]+[@＠][A-Za-z0-9.-Ａ-Ｚａ-ｚ０-９]+[.．][A-Za-zＡ-Ｚａ-ｚ]{2,}")
# Very simple person name heuristic (kanji/katakana/hiragana + optional space + kanji etc.)
PAT_PERSON = re.compile(r"(?:[一-龥々〆ヵヶ]{2,4}[\s　]?[一-龥々〆ヵヶ]{1,4}|[ァ-ヴー]{2,20}[\s　]?[ァ-ヴー]{2,20}|[ぁ-ん]{2,20}[\s　]?[ぁ-ん]{2,20})(?:さん|様|氏|先生|殿|くん|ちゃん)?")

# Exclusions
PAT_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3])[：:][0-5]\d(?!\d)")
PAT_TIME2 = re.compile(r"(?<!\d)([01]?\d|2[0-3])時([0-5]?\d)?分?")
PAT_MONEY = re.compile(rf"[{DIG},，]+\s*円")
PAT_PCT = re.compile(rf"[{DIG}]+\s*[％%]")
PAT_DATE = re.compile(r"\b\d{4}[./年]\s?\d{1,2}[./月]\s?\d{1,2}(日)?\b")
PAT_DATE2 = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
PAT_UNIT = re.compile(rf"\d+\s*(歳|人|件|回|本|台|円|分|時間)")
NEG_PERSON = [
    "です", "ます", "でした", "いたし", "お願い", "お問い合わせ", "お問い合", "確認", "失礼", "お忙", "連絡",
    "学習", "勉強", "塾", "受験", "対応", "希望", "初め", "検討", "主体", "電話", "ありがとうございました", "ありがとうございます",
    "お世話", "昨日", "時間", "先受験", "選択", "検討中", "次第", "かしこまりました"
]
HONORIFICS = ["さん", "様", "氏", "先生", "殿", "くん", "ちゃん"]
NAME_LABEL_LEFT = ["氏名", "お名前", "名前", "名は", "お子様のお名前", "担当", "ご担当", "名義", "申込者", "と申", "申します", "申す"]


def _not_noise(span: str) -> bool:
    return not (PAT_TIME.search(span) or PAT_TIME2.search(span) or PAT_MONEY.search(span) or PAT_PCT.search(span) or PAT_DATE.search(span) or PAT_DATE2.search(span) or PAT_UNIT.search(span))


def mask_text(text: str, tokens: MaskTokens, precision: str = "normal", entities: dict = None) -> str:
	"""
	precision: "strict" (厳密), "normal" (標準), "loose" (緩い)
	entities: {"address": bool, "postal": bool, "phone": bool, "email": bool, "person": bool}
	"""
	if entities is None:
		entities = {"address": True, "postal": True, "phone": True, "email": True, "person": True}
	
	out = text
	# Order: address -> postal -> phone -> email -> person
	
	if entities.get("address", True):
		out = PAT_ADDRESS.sub(tokens.ADDRESS, out)
	
	if entities.get("postal", True):
		out = PAT_POSTAL.sub(tokens.POSTAL, out)
	
	if entities.get("phone", True):
		out = PAT_PHONE.sub(tokens.PHONE, out)

	# Email with noise guard
	if entities.get("email", True):
		def _sub_email(m: re.Match) -> str:
			span = m.group(0)
			return tokens.EMAIL if _not_noise(span) else span
		out = PAT_EMAIL.sub(_sub_email, out)

	# Person with strict context/honorific guard
	if entities.get("person", True):
		def _sub_person(m: re.Match) -> str:
			s = m.string
			start, end = m.start(), m.end()
			span = m.group(0)
			# basic guards
			if any(kw in span for kw in NEG_PERSON):
				return span
			# length
			if not (2 <= len(span) <= 30):
				return span
			
			# Precision-based masking
			if precision == "strict":
				# 厳密: 敬語または左側のラベルがある場合のみマスク
				right = s[end:min(len(s), end + 8)]
				if any(h in span for h in HONORIFICS) or any(h in right for h in HONORIFICS):
					left = s[max(0, start - 12):start]
					if any(lbl in left for lbl in NAME_LABEL_LEFT):
						return tokens.DEFAULT
					return tokens.DEFAULT
				left = s[max(0, start - 12):start]
				if any(lbl in left for lbl in NAME_LABEL_LEFT):
					return tokens.DEFAULT
				return span
			elif precision == "loose":
				# 緩い: より多くのパターンをマスク
				right = s[end:min(len(s), end + 12)]
				if any(h in span for h in HONORIFICS) or any(h in right for h in HONORIFICS):
					return tokens.DEFAULT
				left = s[max(0, start - 20):start]
				if any(lbl in left for lbl in NAME_LABEL_LEFT):
					return tokens.DEFAULT
				# 長さが適切で、否定キーワードがなければマスク
				return tokens.DEFAULT
			else:  # normal
				# 標準: 現在のロジック
				right = s[end:min(len(s), end + 8)]
				if any(h in span for h in HONORIFICS) or any(h in right for h in HONORIFICS):
					return tokens.DEFAULT
				left = s[max(0, start - 12):start]
				if any(lbl in left for lbl in NAME_LABEL_LEFT):
					return tokens.DEFAULT
				return span

		out = PAT_PERSON.sub(_sub_person, out)

	return out


if __name__ == "__main__":
	import sys
	tokens = load_tokens()
	
	# Load mask settings from environment
	precision = "normal"
	entities = {"address": True, "postal": True, "phone": True, "email": True, "person": True}
	
	env_settings = os.getenv("MASK_SETTINGS")
	if env_settings:
		try:
			settings = json.loads(env_settings)
			precision = settings.get("precision", "normal")
			if "entities" in settings:
				entities = settings["entities"]
		except Exception:
			pass
	
	data = sys.stdin.read()
	if not data:
		print("", end="")
		sys.exit(0)
	print(mask_text(data, tokens, precision, entities))