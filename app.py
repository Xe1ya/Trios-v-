import concurrent.futures
import streamlit as st
import openai
import anthropic
import google.generativeai as genai

# ページ設定
st.set_page_config(page_title="Trios", page_icon="🔍", layout="centered")

# --- AI取得ロジック ---

def fetch_openai(query, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}]
        )
        return {"provider": "OpenAI (GPT-4o)", "text": response.choices[0].message.content}
    except Exception as e:
        return {"provider": "OpenAI", "text": f"Error: {str(e)}"}

def fetch_anthropic(query, api_key):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            messages=[{"role": "user", "content": query}]
        )
        return {"provider": "Anthropic (Claude 3.5 Sonnet)", "text": message.content[0].text}
    except Exception as e:
        return {"provider": "Anthropic", "text": f"Error: {str(e)}"}

def fetch_google(query, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(query)
        return {"provider": "Google (Gemini 1.5 Pro)", "text": response.text}
    except Exception as e:
        return {"provider": "Google", "text": f"Error: {str(e)}"}

# --- UI構築 ---

def main():
    st.title(" Trios ")
    
    with st.sidebar:
        st.header("⚙️ API Keys")
        o_key = st.text_input("OpenAI Key", type="password")
        a_key = st.text_input("Anthropic Key", type="password")
        g_key = st.text_input("Google Key", type="password")

    query = st.text_input("検索クエリを入力")
    
    if st.button("検索"):
        if not (o_key or a_key or g_key):
            st.error("少なくとも1つのAPIキーを入力してください")
            return
            
        with st.spinner("AIが回答を生成中..."):
            tasks = []
            if o_key:
                tasks.append((fetch_openai, (query, o_key)))
            if a_key:
                tasks.append((fetch_anthropic, (query, a_key)))
            if g_key:
                tasks.append((fetch_google, (query, g_key)))
            
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

