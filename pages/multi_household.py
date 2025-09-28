
# -*- coding: utf-8 -*-
import io
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import pickle as pkl
import base64
from io import BytesIO
from streamlit.components.v1 import html as st_html


from utils import load_model, read_table, feature_names_from_model, cast_inputs, predict_with_model, is_classifier

def show():
    #---------------------------------------------------------------------------------------------
    #كود لحفظ الجلسة
    # مفاتيح session_state
    MODEL_SESSION_KEY = "sess_model_b64"
    MODEL_NAME_KEY = "sess_model_name"
    DATA_SESSION_KEY = "sess_data_b64"
    DATA_NAME_KEY = "sess_data_name"

    # 2) دوال مساعدة لحفظ/استرجاع Base64 في session_state
    def store_file_to_session_state(file_obj, sess_b64_key, sess_name_key):
        """
        يقرأ الملف (UploadedFile أو ملف-like) ويخزن base64 + name في st.session_state.
        """
        try:
            file_obj.seek(0)
            raw = file_obj.read()
            b64 = base64.b64encode(raw).decode()
            st.session_state[sess_b64_key] = b64
            st.session_state[sess_name_key] = getattr(file_obj, "name", "uploaded_file")
        except Exception as e:
            st.error(f"فشل تخزين الملف في الجلسة: {e}")

    def bytesio_from_session(sess_b64_key, sess_name_key):
        """
        إذا وُجد base64 في session_state يعيد BytesIO مع الخاصية name.
        """
        if sess_b64_key in st.session_state:
            try:
                b64 = st.session_state[sess_b64_key]
                raw = base64.b64decode(b64)
                bio = BytesIO(raw)
                bio.name = st.session_state.get(sess_name_key, "restored_file")
                bio.seek(0)
                return bio
            except Exception as e:
                st.warning(f"فشل استعادة الملف من الجلسة: {e}")
        return None

    #---------------------------------------------------------------------------------------------
    st.set_page_config(page_title="تنبؤ دفعي (ملف Excel/CSV)", page_icon="📊", layout="wide")

    st.title("📊 تنبؤ لعدد كبيرة من الأسر على ملف Excel/CSV")
    st.caption("ارفع النموذج ثم الملف؛ سنضيف عموداً جديداً بالتنبؤات مع خيار تنزيله.")

    with st.sidebar:
        st.header("إعداد النموذج")
        model_file = st.file_uploader("ارفع ملف النموذج (.pkl / .joblib)", type=["pkl", "joblib"],  key="uploader_model")
        st.header("الإعدادات")
        pred_col_name = st.text_input("اسم عمود المخرجات", value="prediction")
        add_proba = st.checkbox("إضافة عمود/أعمدة الاحتمالات (للتصنيف)", value=True)
        st.caption("إذا كان النموذج تصنيفياً ويدعم predict_proba، سيتم إلحاق الأعمدة 'proba_*'.")
    #---------------------------------------------------------------------------------------------
    #تبع الجلسة

    if model_file is not None:
        if (MODEL_NAME_KEY not in st.session_state) or (st.session_state.get(MODEL_NAME_KEY) != getattr(model_file, "name", "")):
            store_file_to_session_state(model_file, MODEL_SESSION_KEY, MODEL_NAME_KEY)
            #st.success("تم حفظ ملف النموذج في الجلسة ✅")

    result_model_file = None

    if model_file is None:
        restored = bytesio_from_session(MODEL_SESSION_KEY, MODEL_NAME_KEY)
        if restored:
            result_model_file = restored
    else:
    
        result_model_file = model_file


    model_file = result_model_file

    #---------------------------------------------------------------------------------------------
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
                feature_order = obj["feature_order"]  # ترتيب الأعمدة
                metadata = obj.get("metadata", {})
            
            else:
                model = obj
                feature_order = None

            st.success("تم تحميل النموذج بنجاح ✅")
        except Exception as e:
            st.error(f"تعذر تحميل النموذج: {e}")



    uploaded = st.file_uploader("ارفع ملف البيانات (Excel/CSV)", type=["xlsx", "xls", "csv"], key="uploader_data")

    #---------------------------------------------------------------------------------------------
    #تبع الجلسة
    if uploaded is not None:
        if (DATA_NAME_KEY not in st.session_state) or (st.session_state.get(DATA_NAME_KEY) != getattr(uploaded, "name", "")):
            store_file_to_session_state(uploaded, DATA_SESSION_KEY, DATA_NAME_KEY)
            #st.success("تم حفظ ملف البيانات في الجلسة ✅")

    result_uploaded = None

    if uploaded is None:
        restored2 = bytesio_from_session(DATA_SESSION_KEY, DATA_NAME_KEY)
        if restored2:
            result_uploaded = restored2
    else:
        result_uploaded = uploaded

    uploaded = result_uploaded
    #---------------------------------------------------------------------------------------------

    if uploaded is None:
        st.stop()

    try:
        df = read_table(uploaded)
    except Exception as e:
        st.error(f"تعذر قراءة الملف: {e}")
        st.stop()

    st.write("حجم البيانات:", df.shape)
    st.dataframe(df.head(20))

    # تحديد الأعمدة المدخلة
    expected = None
    if model is not None:
        expected = feature_names_from_model(model)

        

    st.subheader("🔧 اختيار الأعمدة المستخدمة في التنبؤ")
    if expected is not None:
        default_cols = [c for c in df.columns if c in expected]
        help_txt = "تم اقتراح الأعمدة المتوقعة من النموذج. يمكنك تعديلها."
    else:
        default_cols = list(df.columns)
        help_txt = "لم نستطع استنتاج أعمدة محددة؛ افتراضياً سنستخدم جميع الأعمدة."

    selected_cols = st.multiselect("الأعمدة الداخلة إلى النموذج:", options=list(df.columns), default=default_cols, help=help_txt)

    if not selected_cols:
        st.warning("الرجاء اختيار أعمدة على الأقل.")
        st.stop()

    X = df[selected_cols].copy()

    # نحاول تحويل الأنواع: سنفترض جميع المختارة رقمية إن أمكن، والباقي كنص
    schema = [{"name": c, "type": "numeric" if pd.api.types.is_numeric_dtype(X[c]) else "text"} for c in selected_cols]
    X_cast = cast_inputs(X, schema)

    if st.button("تنفيذ التنبؤ وإضافة العمود ✅", use_container_width=True):
        if model is None:
            st.error("الرجاء رفع ملف النموذج أولاً.")
            st.stop()
        try:
            # ✅ تجهيز البيانات (إضافة الأعمدة الناقصة)
            def prepare_input(user_df, feature_order):
                missing_cols = [col for col in feature_order if col not in user_df.columns]
                for col in missing_cols:
                    user_df[col] = np.nan
                # عرض الأعمدة الناقصة للمستخدم (اختياري)
                if missing_cols:
                    st.warning(f"تمت إضافة الأعمدة الناقصة التالية كـ NaN: {missing_cols}")
                # إعادة الترتيب
                return user_df[feature_order]

            # جهز البيانات
            X_ready = prepare_input(X_cast, feature_order)

            # نفذ التنبؤ
            out = predict_with_model(model, X_ready)

            y_pred = out["y_pred"]
            out_df = df.copy()
            out_df[pred_col_name] = y_pred

            if add_proba and "y_proba" in out:
                proba = out["y_proba"]
                class_names = out.get("class_names")
                if class_names is None:
                    class_cols = [f"proba_{i}" for i in range(proba.shape[1])]
                else:
                    class_cols = [f"proba_{str(c)}" for c in class_names]
                proba_df = pd.DataFrame(proba, columns=class_cols, index=out_df.index)
                out_df = pd.concat([out_df, proba_df], axis=1)

            st.success("تم حساب التنبؤات وإضافتها للجدول ✅")
            st.dataframe(out_df.head(50))

        except Exception as e:
            st.error(f"❌ خطأ أثناء التنبؤ: {e}")


            # تنزيل كملف Excel
            output_buf = io.BytesIO()
            with pd.ExcelWriter(output_buf, engine="openpyxl") as writer:
                out_df.to_excel(writer, index=False, sheet_name="predictions")
            output_buf.seek(0)
            st.download_button(
                label="⬇️ تنزيل الملف (Excel)",
                data=output_buf,
                file_name="predictions_with_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"حصل خطأ أثناء التنبؤ: {e}")