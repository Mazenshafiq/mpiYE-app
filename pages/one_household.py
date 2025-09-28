
# -*- coding: utf-8 -*-
import json
import pandas as pd
import streamlit as st
import joblib
import pickle as pkl

from utils import load_model, read_table, feature_names_from_model, infer_schema_from_sample, cast_inputs, predict_with_model

def show():
    st.set_page_config(page_title="تنبؤ صف مفرد", page_icon="🔢", layout="wide")

    st.title("🔢 تنبؤ أسرة واحدة")
    st.caption("ادخل بيانات أسرة واحدة واحصل على التنبؤ فوراً")

    with st.sidebar:
        st.header("إعداد النموذج")
        model_file = st.file_uploader("ارفع ملف النموذج (.pkl / .joblib)", type=["pkl", "joblib"])

        #st.header("مخطط الميزات (اختياري)")
        #schema_file = st.file_uploader("ارفع ملف JSON لمخطط الميزات", type=["json"], help="شكل الملف: [{'name':'age','type':'integer'}, {'name':'sex','type':'category','choices':['M','F']}, ...]")
        #st.caption("أو ارفع **عينة بيانات** لاستنتاج الأعمدة تلقائياً")
        #sample_file = st.file_uploader("عينة بيانات (Excel/CSV)", type=["xlsx", "xls", "csv"])

    # تحميل النموذج
    model = None
    pipeline = None

    if model_file is not None:
        try:
            # لا تستخدم .read() مع pickle/joblib، مرّر كائن الملف مباشرة
            name = model_file.name.lower()
            model_file.seek(0)  # تأكد أن مؤشر القراءة في البداية

            if name.endswith((".joblib",)):
                obj = joblib.load(model_file)
            else:
                obj = pkl.load(model_file)

            # لو كنتَ قد حفظتَ قاموسًا {'pipeline': ...}
            if isinstance(obj, dict) and "pipeline" in obj:
                pipeline = obj["pipeline"]
                model = pipeline
            else:
                model = obj

            st.success("تم تحميل النموذج بنجاح ✅")
        except Exception as e:
            st.error(f"تعذر تحميل النموذج: {e}")


    # تجهيز المخطط (schema)
    schema = None
    expected_cols = None
    if model is not None:
        expected_cols = feature_names_from_model(model)

    #if schema_file is not None:
        #try:
            #schema = json.load(schema_file)
        #except Exception as e:
            #st.error(f"تعذر قراءة ملف المخطط JSON: {e}")

    sample_df = None
    #if sample_file is not None:
        #try:
            #sample_df = read_table(sample_file)
            #st.caption(f"تم تحميل عينة بحجم: {sample_df.shape}")
        #except Exception as e:
            #st.error(f"تعذر قراءة العينة: {e}")

    if schema is None:
        if sample_df is not None:
            schema = infer_schema_from_sample(sample_df, expected_cols)
        elif expected_cols is not None:
            # نصنع مخططاً بسيطاً من أسماء الأعمدة فقط (نوع نصي افتراضياً)
            schema = [{"name": c, "type": "numeric"} for c in expected_cols]

    if schema is None:
        st.warning("🔧 لبدء الإدخال: ارفع **ملف النموذج** ثم **مخطط الميزات** أو **عينة بيانات** لاكتشاف الأعمدة.")
        st.stop()



    st.subheader("🧾 أدخل القيم")


    cols = st.columns(6)
    values = {}
    for i, item in enumerate(schema):
        name = item["name"]
        typ = item.get("type", "text")
        choices = item.get("choices", None)
        with cols[i % 6]:
            if typ == "integer":
                values[name] = st.number_input(f"{name}", value=0, step=1)
            elif typ == "numeric":
                #values[name] = st.number_input(f"{name}", value=0.0, format="%.6f")
                values[name] = st.number_input(f"{name}", value=0)
            elif typ == "category" and choices:
                values[name] = st.selectbox(f"{name}", choices=choices)
            else:
                values[name] = st.text_input(f"{name}", value="")

    # تحويل إلى DataFrame والأنواع
    row_df = pd.DataFrame([values])
    row_df = cast_inputs(row_df, schema)

    do_predict = st.button("تنفيذ التنبؤ الآن ✅", use_container_width=True)

    if do_predict:
        if model is None:
            st.error("الرجاء رفع النموذج أولاً.")
            st.stop()
        try:
            out = predict_with_model(model, row_df)
            y_pred = out.get("y_pred")
            st.success(f"الناتج: **{y_pred[0]}**")
            if "y_proba" in out:
                proba = out["y_proba"]
                class_names = out.get("class_names")
                st.write("احتمالات الأصناف:")

    # عرض كل صنف باحتماله مع progress bar
                #for i, p in enumerate(proba):
                    #label = class_names[i] if class_names is not None else f"Class {i}"
                # st.write(f"{label}: {p:.4f}")
                # st.progress(float(p))

                # جدول الاحتمالات أيضًا (للمقارنة الرقمية)
                #proba_df = pd.DataFrame([proba], columns=[str(c) for c in (class_names if class_names is not None else range(len(proba)))])
                #st.dataframe(proba_df.style.format("{:.4f}"))

                proba_df = pd.DataFrame(proba, columns=[str(c) for c in (class_names if class_names is not None else range(proba.shape[1]))])
                st.dataframe(proba_df.style.format("{:.3f}"))
            with st.expander("عرض بيانات الإدخال كما استقبلها النموذج"):
                st.dataframe(row_df)
        except Exception as e:
            st.error(f"حصل خطأ أثناء التنبؤ: {e}")
