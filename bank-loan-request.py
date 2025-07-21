import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import pandas as pd
import numpy as np

# من اجل تنسيق اللغة العربية
import arabic_reshaper
from bidi.algorithm import get_display


# ================================
#  تحميل النموذج بفعالية
@st.cache_resource

# تحميل النموذج والبيانات
def load_model(path="Bmodel.pkl"):
    with open(path, "rb") as file:
        saved = pickle.load(file)
    return saved["model"], saved["X_train"], saved["y_train"]

#  تحميل البيانات
@st.cache_data
def load_data(path="d:/CSV_FILLS/loan_data_set.csv"):
    return pd.read_csv(path)

#  رسم المخططات البيانية
def draw_boxplot(df, x, y, title,xlabel,ylabel, palette):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=x, y=y, palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

def draw_countplot(df, x, hue, title,xlabel,ylabel, palette):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=x, hue=hue, palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)


def draw_scatter(df, x, y, hue, title, palette):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def draw_barplot(df, x, y, hue, title,xlabel,ylabel, palette):#palette لاختيار لون الاعمدة
    fig, ax = plt.subplots()
    sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette,ax=ax)
    ax.set_title(title)
    st.pyplot(fig)   
   
def draw_stacked_bar_chart(x, y, title, palette):# hue يتعامل مع بيانات مهيكلة مجمعة ولا يستخدم 
    fig, ax = plt.subplots()
    y.plot(kind="bar", stacked=True, color=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel("النسبة المئوية")
    st.pyplot(fig)
  

def draw_pieplot( x, labels,title,autopct, palette):#palette لاختيار لون الاعمدة
    fig, ax = plt.subplots()
    ax.pie(x=sizes, labels=labels,autopct=autopct, colors=palette)
    ax.set_title(title)
    st.pyplot(fig)

# ================================

#  واجهة المستخدم

st.set_page_config(page_title="نظام التنبؤ بالقرض", layout="wide")
st.markdown("<h1 style='text-align: center; color: #3E8ED0; font-family: Cairo;'>نظام التنبؤ بقرار القرض</h1>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 التنبؤ بالنتيجة", "📈 التحليل البياني","📊 مؤشرات الأداء ","ℹ️ معلومات", ])

df = load_data()
model, target_col, target_values = load_model()


# ================================
# تبويب التنبؤ بالنتيجة
with tab1:
    
#  مدخلات العميل

    st.subheader("📝 بيانات العميل")
    Gender = st.selectbox("الجنس", ["ذكر", "أنثى"])
    Married = st.selectbox("هل متزوج؟", ["نعم", "لا"])
    Dependents = st.selectbox("عدد المعالين", ["0", "1", "2", "3+"])
    Education = st.selectbox("المستوى التعليمي", ["جامعي", "ثانوي"])
    Self_Employed = st.selectbox("هل يعمل لحسابه؟", ["نعم", "لا"])
    ApplicantIncome = st.number_input("دخل مقدم الطلب")
    CoapplicantIncome = st.number_input("دخل المتشارك")
    LoanAmount = st.number_input("مبلغ القرض المطلوب")
    Loan_Amount_Term = st.number_input("مدة سداد القرض (بالأشهر)")
    Credit_History = st.selectbox("السجل الائتماني", ["0", "1"])
    Property_Area = st.selectbox("منطقة السكن", ["حضري", "نصف حضري", "ريفي"])
  

    #تحويل البيانات
    if st.button("🔍 تنبؤ بالنتيجة"):
        input_case = [[
            int(1 if Gender == "ذكر" else 0),
            int(1 if Married == "نعم" else 0),
            int(Dependents) if str(Dependents).isdigit() else 3,  # معالجة "3+"
            int(1 if Education == "جامعي" else 0),
            int(1 if Self_Employed == "نعم" else 0),
            float(ApplicantIncome),
            float(CoapplicantIncome),
            float(LoanAmount),
            float(Loan_Amount_Term),
            int(Credit_History),
            int({"ريفي": 0, "نصف حضري": 1, "حضري": 2}[Property_Area])
        ]]

        # columns = [ "Gender", "Married", "Dependents", "Education", "Self_Employed",
        #             "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        #             "Credit_History", "Property_Area"]

        # input_df = pd.DataFrame(input_case, columns=columns)
        try:
            prediction = model.predict(input_case)[0]
            probas = model.predict_proba(input_case)[0]

            result_text = "✔️ مؤهل للحصول على قرض" if prediction == 1 else "❌ غير مؤهل للحصول على قرض"
            st.markdown(f"""
            <div style='border:2px solid #3E8ED0; padding: 15px; border-radius:10px; background-color: #F0F8FF;'>
            <h4 style='color:#2C3E50;'>النتيجة:</h4>
            <p style='font-size:18px; color:#27AE60;'>{result_text}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"📊 نسبة القبول: **{probas[1]*100:.1f}%**")
            st.markdown(f"📉 نسبة الرفض: **{probas[0]*100:.1f}%**")
        except Exception as e:
            st.warning(f"حدث خطأ أثناء التنبؤ: {e}")


with tab2:

#المخطط الأول
    with st.expander("📦 توزيع دخل مقدم الطلب حسب حالة القرض"):
            
            text = "دخل مقدم الطلب حسب حالة القرض"
            reshaped = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped)

            draw_boxplot(df, "Loan_Status", "ApplicantIncome",bidi_text,"Loan_Status", "ApplicantIncome", "Set2")
            st.markdown("### 📋  التفاصيل الرقمية والتحليل:")
            st.write(df.groupby("Loan_Status")["ApplicantIncome"].describe())

            st.markdown("""
            <div style='background-color:#F0F8FF; border-left:5px solid #3E8ED0; padding:10px; margin-top:10px; font-family:Cairo; font-size:16px;'>
            💡 <strong>التحليل والاستنتاج:</strong> 
                    
                        ✔️ لا يوجد فرق ذو دلالة إحصائية واضحة بين متوسط الدخل في الحالتين.
                   
                        ✔️ الوسيط تقريبًا نفسه، مما يعزز فكرة أن الدخل لا يؤثر بشكل مباشر على قرار القرض.
                    
                        🔍 التفسير المنطقي:
                            ما نراه هنا يشير إلى أن قرار القرض لا يعتمد فقط على الدخل، بل ربما يعتمد أكثر على متغيرات أخرى مثل:
                            التاريخ الائتماني (Credit History)
                            مستوى التعليم
                            الوظيفة أو نوع العمل
                            مبلغ القرض المطلوب
            """, unsafe_allow_html=True)


#المخطط الثاني
    with st.expander("📦 توزيع دخل المتشارك حسب حالة القرض"):
            
            text = "دخل المتشارك حسب حالة القرض"
            reshaped = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped)

            draw_boxplot(df, "Loan_Status", "CoapplicantIncome",bidi_text,"Loan_Status", "CoapplicantIncome", "Set3")
            st.markdown("### 📋  التفاصيل الرقمية والتحليل:")
            st.write(df.groupby("Loan_Status")["CoapplicantIncome"].describe())

            st.markdown("""
            <div style='background-color:#F0FFF0; border-left:5px solid #27AE60; padding:10px; margin-top:10px; font-family:Cairo; font-size:16px;'>
            💡 <strong>تحليل:</strong> دخل المتشارك يُظهر تأثيرًا محدودًا، لكن يمكن أن يعزز فرص الموافقة عندما يكون دخل مقدم الطلب منخفضًا.
            </div>
            """, unsafe_allow_html=True)


#المخطط الثالث
    with st.expander("📊 عدد الطلبات حسب منطقة السكن وحالة القرض"):

        text = "عدد الطلبات حسب منطقة السكن"
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)

        draw_countplot(df, "Property_Area", "Loan_Status", bidi_text,"Property_Area", "count", "pastel")
        st.markdown("  🏙️ Urban – حضري   🌆 Semiurban – نصف حضري  🌾 Rural – ريفي")
        st.markdown("### 📋  التفاصيل الرقمية والتحليل:")
        st.write(df.groupby(["Property_Area", "Loan_Status"]).size().unstack())
     
        
        st.markdown("""
        <div style='background-color:#FFF8E1; border-left:5px solid #FFA500; padding:10px; margin-top:10px; font-family:Cairo; font-size:16px;'>
        💡 <strong>تحليل:</strong> المنطقة النصف حضرية تُسجل أعلى عدد من الطلبات والموافقات، مما يعكس تفضيل البنوك للعملاء القاطنين في مناطق ذات نشاط اقتصادي متوسط.
        </div>
        """, unsafe_allow_html=True)

#المخطط الرابع
    with st.expander("📈 العلاقة بين مبلغ القرض والدخل"):

        text ="العلاقة بين مبلغ القرض والدخل"
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
    
        draw_scatter(df, "ApplicantIncome", "LoanAmount", "Loan_Status",bidi_text , "coolwarm")
        st.markdown("### 📋  التفاصيل الرقمية والتحليل:")
        st.write(f"📌 معامل الارتباط: {df[['ApplicantIncome', 'LoanAmount']].corr().iloc[0,1]:.2f}")
        st.markdown("""
        <div style='background-color:#E8F8F5; border-left:5px solid #1ABC9C; padding:10px; margin-top:10px; font-family:Cairo; font-size:16px;'>
        💡 <strong>تحليل:</strong> هناك ارتباط طردي واضح بين دخل العميل ومبلغ القرض، ما يُشير إلى أن الدخل المرتفع يُفسح المجال لطلب مبالغ أعلى بثقة.
        </div>
        """, unsafe_allow_html=True)

#المخطط الخامس       
    with st.expander("📉 نسب الموافقة حسب السجل الائتماني"):

        credit_counts = df.groupby(["Credit_History", "Loan_Status"]).size().unstack()
        credit_rates = credit_counts.div(credit_counts.sum(axis=1), axis=0)
     
        text = "نسبة الموافقة حسب السجل الائتماني"
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
    
        draw_stacked_bar_chart("Credit_History", credit_rates,bidi_text, ["#E74C3C", "#2ECC71"])
        st.markdown("### 📋  التفاصيل الرقمية والتحليل:")
        st.write(credit_rates)  

        st.markdown("""
        <div style='background-color:#FDEDEC; border-left:5px solid #C0392B; padding:10px; margin-top:10px; font-family:Cairo; font-size:16px;'>
        💡 <strong>تحليل:</strong> أصحاب السجل الائتماني الإيجابي (1) يتمتعون بنسبة قبول تفوق 80٪، مما يُبرز السجل كعامل رئيسي مؤثر في قرار القرض.
        </div>
        """, unsafe_allow_html=True)


#المخطط السادس
    with st.expander("📊 مخطط النِسب المئوية حسب الحالة الاجتماعية "):
        
        text = "النِسب المئوية حسب الحالة الاجتماعية"
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)

        draw_countplot(df, "Married", "Loan_Status",bidi_text,"Marital Status","Number of Applicants",
                                                                                                            "muted")
        
        st.markdown("### 📋  التفاصيل الرقمية والتحليل:")
        marital_counts = df.groupby(["Married", "Loan_Status"]).size().unstack()
        st.write(marital_counts)
      
        st.markdown("""
        <div style='background-color:#FEF9E7; border-left:5px solid #F39C12; padding:10px; margin-top:10px; font-family:Cairo; font-size:16px;'>
        💡 <strong>تحليل:</strong> الفئة "متزوج" تُسجل معدل موافقة أعلى، وقد يُعتبر ذلك مؤشرًا على الاستقرار المالي والاجتماعي في نظر البنوك.
        </div>
        """, unsafe_allow_html=True)

#المخطط السابع
    with st.expander("📊 نسبة الموافقة حسب المستوى التعليمي "):

        text = " نسبة الموافقة حسب المستوى التعليمي "
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)

       
        draw_countplot(df, "Education", "Loan_Status",bidi_text,"Education Level","Number of Applications","Blues")
        st.markdown("### 📋  التفاصيل الرقمية والتحليل:")
        edu_counts = df.groupby(["Education", "Loan_Status"]).size().unstack()
        st.write(edu_counts)
        st.markdown("""
        <div style='background-color:#EBF5FB; border-left:5px solid #3498DB; padding:10px; margin-top:10px; font-family:Cairo; font-size:16px;'>
        💡 <strong>تحليل:</strong> الطلبات من أصحاب التعليم الجامعي تحظى بموافقة أكبر، مما يُشير إلى أن التعليم يُعد مؤشرًا على القدرة المالية والانضباط.
        </div>
        """, unsafe_allow_html=True)

#المخطط الثامن
    with st.expander("📊 Pie Chart لنسب القبول الإجمالية "):
        
     
        text = "نسب القبول الإجمالية"
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)

        labels = df["Loan_Status"].value_counts().index
        sizes = df["Loan_Status"].value_counts().values
        draw_pieplot(sizes, labels,bidi_text, '%1.1f%%',["#2ECC71", "#E74C3C"])
          
        st.markdown("### 📋  التفاصيل الرقمية والتحليل:")
        total_counts = df["Loan_Status"].value_counts(normalize=True) * 100
        st.write(total_counts.round(1).astype(str) + ' %')
        st.markdown("""
        <div style='background-color:#F8F9F9; border-left:5px solid #7F8C8D; padding:10px; margin-top:10px; font-family:Cairo; font-size:16px;'>
        💡 <strong>تحليل:</strong> يظهر أن أكثر من 68   ٪ من الطلبات يتم قبولها، مما يدل على مرونة النموذج أو سياسات البنك تجاه العملاء المؤهلين.
        </div>
        """, unsafe_allow_html=True)

#المخطط التاسع
    with st.expander("📊 العلاقة بين مدة القرض وحالة القبول"): 

        text = " العلاقة بين مدة القرض وحالة القبول"
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)



        draw_boxplot(df,"Loan_Status","Loan_Amount_Term",bidi_text,
                    "Loan Status","Loan Term (Months)", "Set1")    
        st.markdown("### 📋  التفاصيل الرقمية والتحليل:")
        st.write(df.groupby("Loan_Status")["Loan_Amount_Term"].describe())
        st.markdown("""
        <div style='background-color:#FFF2F0; border-left:5px solid #E67E22; padding:10px; margin-top:10px; font-family:Cairo; font-size:16px;'>
        💡 <strong>تحليل:        </strong> 
                    
                    مدة القرض لا تبدو عاملًا مؤثرًا بشكل واضح في حالة القبول أو الرفض.

                    الوسيط في الحالتين متقارب جدًا (360 شهر)، مما يعني أن مؤسسات القرض تميل لمنح نفس المدة بغض النظر عن حالة القبول.

                    التوزيع المتشابه يشير إلى أن قرار الموافقة على القرض لا يعتمد أساسًا على مدة القرض.
            
        </div>
        """, unsafe_allow_html=True)

# ================================
with tab3:
    st.markdown("<h2 style='color:#3E8ED0;'>📊 مصفوفة الارتباك </h2>", unsafe_allow_html=True)

    # تحميل النموذج وبيانات التدريب
    model, X_train_raw, y_train_raw = load_model()
    X_train = pd.DataFrame(X_train_raw)
    y_train = pd.Series(y_train_raw)

    # التنبؤ
    y_pred = model.predict(X_train)

    # حساب مصفوفة الارتباك
    conf_mat = confusion_matrix(y_train, y_pred, labels=[1, 0])

    st.markdown("تعرض هذه المصفوفة عدد الحالات المصنفة بشكل صحيح أو خاطئ:")

    fig, ax = plt.subplots()
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=[ get_display(arabic_reshaper.reshape("توقع: موافقة")), get_display(arabic_reshaper.reshape("توقع: رفض"))],
                yticklabels=[ get_display(arabic_reshaper.reshape("فعلي: موافقة")),get_display(arabic_reshaper.reshape("فعلي: رفض"))],
                ax=ax)
    ax.set_xlabel( get_display(arabic_reshaper.reshape("توقع النموذج")))
    ax.set_ylabel( get_display(arabic_reshaper.reshape("النتيجة الفعلية")))
    ax.set_title( get_display(arabic_reshaper.reshape(" مصفوفة الارتباك للقروض")))
    st.pyplot(fig)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    acc = accuracy_score(y_train, y_pred)
    prec = precision_score(y_train, y_pred, pos_label=1)
    rec = recall_score(y_train, y_pred, pos_label=1)
    f1 = f1_score(y_train, y_pred, pos_label=1)

    st.markdown("### 📊 مؤشرات الأداء:")
    st.metric("✅ الدقة (Accuracy)", f"{acc*100:.2f}%")
    st.metric("🎯 Precision", f"{prec*100:.2f}%")
    st.metric("📈 Recall", f"{rec*100:.2f}%")
    st.metric("📊 F1 Score", f"{f1*100:.2f}%")

# ================================
# ℹ تبويب المعلومات
with tab4:
     st.markdown("""
        <div style='background-color: #F0F8FF; padding: 15px; border-radius: 10px; border: 1px solid #3E8ED0;'>
        <p style='font-size:18px; font-family: "Cairo", sans-serif; color:#2C3E50;'>
        لتوقع إمكانية الموافقة على طلب القرض بناءً على البيانات المدخلة Naive Bayes  هذا النظام يستخدم نموذج <br>               
        يتم عرض نتائج التوقع، والنسب، بالإضافة إلى تحليل بصري شامل للبيانات الأصلية المستخدمة في تدريب النموذج<br>
        </p>
        </div>
        """, unsafe_allow_html=True)
# ================================
#footer

st.markdown("<p style='text-align: center; color: gray;'>© 2025 |      Streamlit و Naive Bayes مشروع مقدم بواسطة ابراهيم غريب باستخدام نظام التنبؤ </p>", unsafe_allow_html=True)

