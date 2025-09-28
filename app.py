
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from pages import one_household, multi_household, info

# ====== قائمة الصفحات ======
PAGES = {
    "📈 حول": info,
    "📥 تنبأ بأسرة واحدة": one_household,
    "📊 تنبأ بأسر متعددة": multi_household,
}

# ====== التنقل بين الصفحات ======
st.sidebar.title("الصفحات")
choice = st.sidebar.radio("اختر الواجهة:", list(PAGES.keys()))

# ====== عرض الصفحة المختارة ======
page = PAGES[choice]
page.show()


