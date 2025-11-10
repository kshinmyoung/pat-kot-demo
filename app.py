# ===============================
# PATH AI Writing Tutor (Mode-Enhanced Stable)
# ===============================
import os
import re
import csv
import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from rapidfuzz import process, fuzz

# -------------------------------
# 기본 UI 설정
# -------------------------------
st.set_page_config(page_title="PATH AI Writing Tutor", page_icon="🧭", layout="centered")
st.title("🧭 PATH AI writing tutor— 신학 유학생용 한국어 작문 튜터")
st.caption("Pedagogical AI writing tutor for Theology and Humanities (TOPIK 3–4)")

# -------------------------------
# Secrets / .env 로드
# -------------------------------
load_dotenv()  # 로컬 개발 시 사용

def _get_secret(key: str):
    """Streamlit Secrets 우선, 없으면 환경변수 사용"""
    try:
        val = st.secrets.get(key, None)
    except Exception:
        val = None
    if not val:
        val = os.getenv(key)
    if val and not isinstance(val, str):
        val = str(val)
    return val

API_KEY = _get_secret("OPENAI_API_KEY")
ADMIN_CODE_SECRET = _get_secret("ADMIN_CODE")

# 상태 캡션(키 노출 방지 마스킹)
masked = (API_KEY[:6] + "…") if API_KEY else "None"
st.caption(f"🔒 API Key: {'감지됨' if API_KEY else '없음'} ({masked})")
st.caption(f"🛂 Admin Code: {'감지됨' if ADMIN_CODE_SECRET else '없음'}")

# -------------------------------
# OpenAI SDK 안전 초기화
# -------------------------------
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None
    st.error(f"⚠️ OpenAI SDK 로드 실패: {e}")

client = None
if OpenAI:
    try:
        client = OpenAI(api_key=API_KEY) if API_KEY else OpenAI()
    except TypeError:
        st.error("⚠️ OpenAI 초기화(TypeError): 라이브러리 버전 충돌 가능. requirements.txt 버전 고정 후 Rerun 하세요.")
        client = None
    except Exception as e:
        st.error(f"⚠️ OpenAI 초기화 실패: {e}")
        client = None

# -------------------------------
# 상수/유틸
# -------------------------------
LOG_COLUMNS = ["timestamp", "pid", "trial", "lang", "topic", "stage", "text"]

def save_log(pid: str, trial: int, lang: str, topic: str, stage: str, text: str):
    """연구용 로그 저장 (누적 CSV)"""
    if not pid:
        return
    row = [datetime.datetime.now().isoformat(), pid, trial, lang, topic, stage, text]
    with open("logs.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

@st.cache_data(show_spinner=False)
def load_corpus(path: str = "corpus.csv") -> pd.DataFrame:
    """성경 병렬 코퍼스 로드 + 컬럼 표준화"""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # 컬럼 맵핑
    mapping = {}
    want_map = {
        "book": ["book", "책", "성경", "서명"],
        "chapter": ["chapter", "chap", "장"],
        "verse": ["verse", "vr", "절"],
        "ko": ["ko", "kor", "korean", "한글", "본문", "본문(한)"],
        "en": ["en", "eng", "english", "영문", "본문(영)"],
        "tags": ["tags", "tag", "주제", "키워드"],
    }
    for want, cands in want_map.items():
        for c in cands:
            if c in df.columns:
                mapping[c] = want
                break
    df = df.rename(columns=mapping)

    required = ["book", "chapter", "verse", "ko", "en", "tags"]
    missing = [r for r in required if r not in df.columns]
    if missing:
        st.error(f"corpus.csv에 다음 컬럼이 필요합니다: {missing}")
        st.stop()

    for c in ("chapter", "verse"):
        try:
            df[c] = df[c].astype(int)
        except Exception:
            pass
    for c in ("tags", "ko", "en"):
        df[c] = df[c].astype(str).fillna("")
    return df

@st.cache_data(show_spinner=False)
def load_prompt(path: str) -> str:
    """피드백 프롬프트 로드(없으면 안전한 기본 프롬프트 제공)"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return (
            "You are a Korean academic writing tutor for theology students (TOPIK 3–4). "
            "Use short, polite Korean. Provide: Praise → two errors (explain & example) → "
            "two related Bible examples → rewrite instruction → strength & next goal. "
            "Keep within 10–12 lines."
        )

corpus = load_corpus()
prompt_kr = load_prompt("feedback_prompt.txt")
prompt_en = load_prompt("feedback_prompt_en.txt")

# -------------------------------
# 코퍼스 검색/서식
# -------------------------------
def extract_keywords(text: str, topn: int = 6):
    toks = re.findall(r"[가-힣A-Za-z]{2,}", text)
    return toks[:topn] if toks else []

def lookup_examples(text: str, topk: int = 2) -> list[dict]:
    kws = extract_keywords(text)
    if not kws:
        return corpus.sample(n=min(topk, len(corpus))).to_dict(orient="records")
    pool = corpus["tags"].tolist() + corpus["ko"].tolist() + corpus["en"].tolist()
    candidates = []
    for kw in kws:
        m = process.extractOne(kw, pool, scorer=fuzz.partial_ratio, score_cutoff=70)
        if not m:
            continue
        val = m[0]
        row = corpus[
            (corpus["tags"].str.contains(re.escape(kw), na=False))
            | (corpus["ko"] == val) | (corpus["en"] == val)
            | (corpus["ko"].str.contains(re.escape(kw)))
            | (corpus["en"].str.contains(re.escape(kw)))
        ].head(1)
        if not row.empty:
            candidates.append(row.iloc[0].to_dict())
    # 중복 제거 + 부족 시 랜덤 보충
    seen, uniq = set(), []
    for r in candidates:
        key = (r["book"], int(r["chapter"]), int(r["verse"]))
        if key not in seen:
            seen.add(key); uniq.append(r)
        if len(uniq) >= topk:
            break
    if len(uniq) < topk:
        rest = topk - len(uniq)
        uniq.extend(corpus.sample(n=min(rest, len(corpus))).to_dict(orient="records"))
    return uniq[:topk]

def format_bible_examples(rows: list[dict]) -> str:
    out = []
    for r in rows:
        out.append(f"📖 {r['book']} {r['chapter']}:{r['verse']} — {r['ko']}\n({r['en']})")
    return "\n".join(out)

# -------------------------------
# 모드별 시스템/유저 프롬프트 빌더 (강화)
# -------------------------------
def build_system_msg(language: str) -> str:
    if language == "한국어 (KR)":
        return (
            "You are a Korean academic writing tutor for theology students. "
            "Respond ONLY in Korean. Use polite '-습니다' style. "
            "Never include English unless the user text itself is English."
        )
    elif language == "영어 (EN)":
        return (
            "You are an academic writing tutor for theology students. "
            "Respond ONLY in English. Do not include any Korean."
        )
    else:  # 이중언어 (KR+EN)
        return (
            "You are a bilingual (KR+EN) academic writing tutor for theology students. "
            "First, produce a full Korean feedback section. Then add a separator line "
            "and provide a concise English summary (2–3 lines)."
        )

def build_user_prompt(base_prompt: str, language: str, student_text: str,
                      examples_block: str, strategy: str) -> str:
    if language == "한국어 (KR)":
        return f"""
{base_prompt}

[학생의 글]
{student_text}

[관련 성경 예시]
{examples_block}

[교수전략]
{strategy}

[출력 형식 엄수]
- 반드시 **한국어**로만 작성
- 10~12줄, '-습니다'체
- 구조: 칭찬 → 오류2(설명+고친예) → 성경예시 요약2 → 재작성 지시 → 강점/다음목표
"""
    elif language == "영어 (EN)":
        return f"""
{base_prompt}

[STUDENT TEXT]
{student_text}

[RELATED BIBLE EXAMPLES]
{examples_block}

[INSTRUCTIONAL STRATEGY]
{strategy}

[OUTPUT FORMAT - STRICT]
- Respond **ONLY in English**
- 8–10 lines, academic tone
- Structure: Praise → 2 errors (explain+example) → 2 Bible examples (brief) → Rewrite instruction → Strength/Next goal
"""
    else:  # 이중언어
        return f"""
{base_prompt}

[학생의 글 / Student Text]
{student_text}

[관련 성경 예시 / Bible Examples]
{examples_block}

[교수전략 / Strategy]
{strategy}

[OUTPUT FORMAT - STRICT]
(1) [KR] 한국어 섹션 (10~12줄, '-습니다'체)
    - 칭찬 → 오류2(설명+고친예) → 성경예시 요약2 → 재작성 지시 → 강점/다음목표
(2) ----------  ← 이 구분선 반드시 포함
(3) [EN] English brief (2–3 lines)
    - Summarize key fixes and rewrite goal
"""

# -------------------------------
# 출력 검증기(모드 위반 자동 안내)
# -------------------------------
def validate_output_by_mode(output: str, language: str) -> str:
    kr = len(re.findall(r"[가-힣]", output))
    en = len(re.findall(r"[A-Za-z]", output))

    if language == "한국어 (KR)":
        if en > kr * 0.2:
            output = "⚠️ (자동 점검) 영어 비율이 높습니다. 한국어로만 간결하게 작성해 주세요.\n\n" + output
    elif language == "영어 (EN)":
        if kr > en * 0.2:
            output = "⚠️ (Auto check) Too much Korean detected. Respond in English only.\n\n" + output
    else:  # 이중언어
        if "----------" not in output or "[EN]" not in output:
            output += "\n\n----------\n[EN] Please add a 2–3 line English summary of key feedback and rewrite goal."
    return output

# -------------------------------
# 사이드바 (참가자 & 교수자)
# -------------------------------
st.sidebar.header("참여자")
pid = st.sidebar.text_input("이니셜/참가자코드 (예: S01)")
trial = st.sidebar.number_input("시도 회차", min_value=1, value=1, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("교수자 모드")
admin_input = st.sidebar.text_input("접근 코드 입력", type="password")
is_admin = False
if ADMIN_CODE_SECRET and admin_input == ADMIN_CODE_SECRET:
    is_admin = True
elif ADMIN_CODE_SECRET is None and admin_input.strip():
    # 로컬 개발 시 임시 허용 (Cloud에서는 Secrets 설정 권장)
    is_admin = True

if is_admin:
    st.sidebar.success("교수자 모드 활성화 ✅")
    # 로그 미리보기/다운로드/백업/초기화
    if os.path.exists("logs.csv"):
        try:
            df_logs = pd.read_csv("logs.csv", names=LOG_COLUMNS)
            st.sidebar.caption("logs.csv (최근 50행)")
            st.sidebar.dataframe(df_logs.tail(50), use_container_width=True, height=240)
        except Exception:
            st.sidebar.info("(로그 파싱 불가 — raw 다운로드)")
        with open("logs.csv", "rb") as f:
            st.sidebar.download_button("⬇️ logs.csv 내려받기", data=f.read(), file_name="logs.csv", mime="text/csv")
        st.sidebar.markdown("—")
        if st.sidebar.button("🗂 로그 백업"):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"logs_backup_{ts}.csv"
            with open("logs.csv", "rb") as f:
                data = f.read()
            with open(name, "wb") as w:
                w.write(data)
            st.sidebar.success(f"백업 완료: {name}")
        if st.sidebar.button("🧹 로그 초기화(되돌릴 수 없음)"):
            open("logs.csv", "w", encoding="utf-8").close()
            st.sidebar.warning("logs.csv 초기화 완료")
            st.experimental_rerun()
    else:
        st.sidebar.info("아직 logs.csv가 없습니다. 첫 사용 시 자동 생성됩니다.")

    # (옵션) 임시 API 키 주입 — Secrets 문제 시 수업 살리기
    if OpenAI and client is None:
        st.sidebar.warning("임시 우회: API 키를 세션에서만 사용 가능")
        temp_key = st.sidebar.text_input("임시 OPENAI_API_KEY", type="password")
        if temp_key.strip():
            try:
                client = OpenAI(api_key=temp_key.strip())
                st.sidebar.success("임시 키 적용됨 (세션 한정)")
            except Exception as e:
                st.sidebar.error(f"임시 키 적용 실패: {e}")

# -------------------------------
# 진단 패널 (교수자 전용만 표시)
# -------------------------------
if is_admin:
    with st.expander("🧑‍💻 관리자용 진단 도구"):
        files = ["app.py", "requirements.txt", "corpus.csv", "feedback_prompt.txt", "feedback_prompt_en.txt"]
        exists = {f: ("✅" if os.path.exists(f) else "❌") for f in files}
        st.table({"파일": list(exists.keys()), "존재": list(exists.values())})

        try:
            df_probe = pd.read_csv("corpus.csv").head(2)
            st.write("corpus.csv 미리보기:", df_probe)
        except Exception as e:
            st.error(f"corpus.csv 읽기 오류: {e}")

        for p in ["feedback_prompt.txt", "feedback_prompt_en.txt"]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    st.write(f"{p} OK (미리보기):", f.read(120) + "…")
            except Exception as e:
                st.error(f"{p} 읽기 오류: {e}")

# -------------------------------
# 본문 UI
# -------------------------------
language = st.radio(
    "피드백 언어 모드",
    ["한국어 (KR) — 한국어만", "영어 (EN) — English only", "이중언어 (KR+EN) — KR + EN summary"],
    index=0, horizontal=True
)
# 선택값 표준화
language = language.split(" — ")[0]

topic = st.selectbox("주제(태그)", ["(자동)", "사랑", "믿음", "기도", "감사", "말씀", "권면", "설명", "요약", "적용"])
student_text = st.text_area("✍️ 학생 글(3–8문장 권장)", height=160, placeholder="예) 저는 오늘 말씀을 통해 ...")
strategy = st.selectbox(
    "교수전략(시연 옵션)", ["자동 선택", "모형 제시 (Modeling)", "단계 안내 (Scaffolding)", "확장 유도 (Extension)"], index=1
)
agree = st.checkbox("연구 참여 및 텍스트 익명 저장에 동의합니다.")

col_btn1, col_btn2 = st.columns(2)
run_clicked = col_btn1.button("💬 피드백 받기")
if col_btn2.button("지우기"):
    st.experimental_rerun()

# -------------------------------
# 데모 피드백(오프라인 폴백 규칙 — 모드 차별화)
# -------------------------------
def demo_feedback(text: str, examples_block: str, lang: str) -> str:
    tips = []
    if re.search(r"(하나님|예수|말씀)[^의]", text):
        tips.append(("[조사]", "명사 뒤 ‘의/을/를’을 정확히.", "예) 하나님의 사랑을 배웠습니다."))
    if re.search(r"다[.!]?$", text):
        tips.append(("[격식]", "'-습니다'로 학술 톤.", "예) 배웠습니다 / 느꼈습니다."))
    if len(tips) < 2:
        tips.append(("[연결어]", "이유-결과 연결: 그래서/그러나.", "예) …배웠습니다. 그래서 감사했습니다."))
    if len(tips) < 2:
        tips.append(("[문장 분리]", "긴 문장은 두 문장으로.", "예) 수업이 끝났습니다. 곧 정리했습니다."))

    if lang == "영어 (EN)":
        return "\n".join([
            "Great effort—your faith and intention are clear.",
            "- [Particles] Use '의/을/를' properly. e.g., 하나님의 사랑을 배웠습니다.",
            "- [Polite ending] Use '-습니다' for academic tone.",
            examples_block.strip() or "📖 (No related Bible example)",
            "Please rewrite in 3–5 sentences using the feedback.",
            "Strength: Clear topic | Next goal: particles & polite endings.",
        ])
    elif lang == "이중언어 (KR+EN)":
        kr = [
            "좋은 시도예요. 신앙의 마음이 잘 느껴집니다.",
            "- [조사] '의/을/를'을 정확히 씁니다. 예) 하나님의 사랑을 배웠습니다.",
            "- [격식] '-입니다/-습니다'체 사용.",
            examples_block.strip() or "📖 (관련 성경 예시 없음)",
            "이제 위 내용을 참고해 3–5문장으로 다시 써보세요.",
            "강점: 주제가 분명함 | 다음 목표: 조사·격식 다듬기",
        ]
        en = [
            "----------",
            "[EN] Focus on particles and polite endings.",
            "Rewrite in 3–5 sentences using the feedback."
        ]
        return "\n".join(kr + en)
    else:
        # 한국어 기본
        lines = [
            "좋은 시도예요. 신앙의 마음이 잘 느껴집니다.",
            "- [조사] '의/을/를'을 정확히 씁니다.",
            "- [격식] '-습니다'체로 정리합니다.",
            examples_block.strip() or "📖 (관련 성경 예시 없음)",
            "3–5문장으로 다시 써보세요.",
            "강점: 주제가 분명함 | 다음 목표: 조사·격식 다듬기",
        ]
        return "\n".join(lines)

# -------------------------------
# 실행 로직
# -------------------------------
if run_clicked:
    if not agree:
        st.warning("연구 참여 및 익명 저장에 동의해 주세요.")
    elif not student_text.strip():
        st.warning("학생 글을 입력하세요.")
    else:
        # 초안 로그
        save_log(pid, trial, language, topic, "draft", student_text)

        # 성경 예시 검색 (+주제 강제 적용)
        examples = lookup_examples(student_text, topk=2)
        if topic and topic != "(자동)":
            tagged = corpus[corpus["tags"].str.contains(re.escape(topic), na=False)]
            if not tagged.empty:
                examples = tagged.sample(n=min(2, len(tagged))).to_dict(orient="records")
        examples_block = format_bible_examples(examples)

        # 프롬프트 선택
        if language == "한국어 (KR)":
            base = prompt_kr
        elif language == "영어 (EN)":
            base = prompt_en
        else:
            base = prompt_kr + "\n\n추가: 위 한국어 피드백 끝에 영어로 2–3줄 핵심 요약을 덧붙이세요."

        # 모드 강화 프롬프트 구성
        system_msg = build_system_msg(language)
        user_msg = build_user_prompt(base, language, student_text, examples_block, strategy)

        # 실제 API 호출 또는 데모 폴백
        if client:
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg[:6000]},
                    ],
                    temperature=0.3,
                    top_p=0.9,
                    max_tokens=900,
                )
                feedback = resp.choices[0].message.content.strip()
            except Exception as e:
                st.warning(f"(API 오류로 데모로 전환) {e}")
                feedback = demo_feedback(student_text, examples_block, language)
        else:
            feedback = demo_feedback(student_text, examples_block, language)

        # 모드 출력 검증
        feedback = validate_output_by_mode(feedback, language)

        st.subheader("💬 AI 피드백")
        st.write(feedback)
        save_log(pid, trial, language, topic, "feedback", feedback)

        st.markdown("---")
        revised = st.text_area("✍️ 재작성(3–5문장): 피드백을 반영해 다시 써보세요.", height=140)
        if st.button("✅ 재작성 제출"):
            if revised.strip():
                save_log(pid, trial, language, topic, "revision", revised.strip())
                st.success("재작성 저장 완료 (logs.csv)")
            else:
                st.warning("재작성 문장을 입력하세요.")
