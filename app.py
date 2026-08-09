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
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": query}]
        )
        return {"provider": "Groq (Llama 3.1 70B)", "text": completion.choices[0].message.content}
    except Exception as e:
        return {"provider": "Groq", "text": f"Error: {str(e)}"}

def fetch_cohere(query, api_key):
    try:
        co = cohere.ClientV2(api_key=api_key)
        response = co.chat(
            model="command-r-plus",
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
    
    if st.button("検索"):
        if not (g_key or groq_key or cohere_key):
            st.error("少なくとも1つのAPIキーを入力してください")
            return
            
        with st.spinner("AIが回答を生成中..."):
            tasks = []
            if g_key:
                tasks.append((fetch_google, (query, g_key)))
            if groq_key:
                tasks.append((fetch_groq, (query, groq_key)))
            if cohere_key:
                tasks.append((fetch_cohere, (query, cohere_key)))
            
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(func, *args) for func, args in tasks]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            
            if results:
                tabs = st.tabs([r["provider"] for r in results])
                for tab, r in zip(tabs, results):
                    with tab:
                        st.markdown(r["text"])

if __name__ == "__main__":
    main()
