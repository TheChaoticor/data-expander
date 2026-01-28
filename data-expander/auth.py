import streamlit as st
import time

MOCK_USERS = {
    "admin": "admin",
    "user": "password",
    "demo": "demo"
}

def login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.is_pro = False

    if st.session_state.authenticated:
        return True

    st.markdown("## 🔐 Login to Data Expander")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if username in MOCK_USERS and MOCK_USERS[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                # Grant pro to admin, others need to 'upgrade'
                st.session_state.is_pro = (username == "admin")
                st.success("Logged in successfully!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    return False

def requires_pro(feature_func):
    def wrapper(*args, **kwargs):
        if not st.session_state.get("is_pro", False):
            st.warning("🔒 This feature is available on the Pro plan.")
            if st.button("Upgrade to Pro ($10/mo)"):
                st.session_state.is_pro = True
                st.success("Upgraded! Enjoy Pro features.")
                st.rerun()
        else:
            return feature_func(*args, **kwargs)
    return wrapper
