import concurrent.futures
import streamlit as st
import google.generativeai as genai
from groq import Groq
import cohere

# ページ設定
st.set_page_config(page_title="Trios", page_icon="🔍", layout="centered")

# --- AI取得ロジック ---

def fetch_google(query, api_key):
    try:
        genai.configure(api_key=api_key)
        
        # 利用可能なモデルを自動取得
        target_model = "gemini-3.6-flash"
       
        model = genai.GenerativeModel(target_model)
        response = model.generate_content(query)
        return {"provider": "Google (Gemini)", "text": response.text}
    except Exception as e:
        return {"provider": "Google", "text": f"Error: {str(e)}"}


def fetch_groq(query, api_key):
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": query}]
        )
        return {"provider": "Groq (Llama 3.3 70B)", "text": completion.choices[0].message.content}
    except Exception as e:
        return {"provider": "Groq", "text": f"Error: {str(e)}"}

def fetch_cohere(query, api_key):
    try:
        co = cohere.ClientV2(api_key=api_key)
        response = co.chat(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": query}]
        )
        return {"provider": "Cohere (Command R+)", "text": response.message.content[0].text}
    except Exception as e:
        return {"provider": "Cohere", "text": f"Error: {str(e)}"}

# --- UI構築 ---

def main():
    st.title("Trios")
    
    with st.sidebar:
        st.header("⚙️ API Keys")
        g_key = st.text_input("Google Key (Gemini)", type="password")
        groq_key = st.text_input("Groq Key (Llama)", type="password")
        cohere_key = st.text_input("Cohere Key", type="password")

    query = st.text_input("検索クエリを入力")
    
# 検索ボタンが押された時の処理
if st.button("検索"):
    results = []
    
    # 1. 各AIから回答を取得
    if google_key:
        results.append(fetch_google(query, google_key))
    if groq_key:
        results.append(fetch_groq(query, groq_key))
    if cohere_key:
        results.append(fetch_cohere(query, cohere_key))
    
    # 各AIの回答を表示
    for res in results:
        st.subheader(res["provider"])
        st.write(res["text"])
        st.divider()

    # 2. 全部の回答が揃ったら「比較まとめ」を作成
    if len(results) > 1:
        st.header("📊 各AIの比較まとめ")
        
        # 各AIの回答を1つのテキストに整理
        all_texts = "\n\n".join([f"【{r['provider']}の回答】\n{r['text']}" for r in results])
        summary_prompt = f"以下の複数のAIの回答を比較し、共通点・違い・それぞれの特徴をわかりやすく要約してください。\n\n{all_texts}"
        
        # 最初に設定されているキーを使ってまとめを生成（例: Google）
        if google_key:
            summary = fetch_google(summary_prompt, google_key)
            st.info(summary["text"])
        elif groq_key:
            summary = fetch_groq(summary_prompt, groq_key)
            st.info(summary["text"])
