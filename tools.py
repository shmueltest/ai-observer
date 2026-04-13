import streamlit as st

def add_cache_button():
    """
    Adds a 'Clear Cache' button to the sidebar. 
    Wipes both data and loaded models/resources.
    """
    if st.sidebar.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.sidebar.success("Cache Cleared!")
        st.rerun()

