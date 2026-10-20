        st.subheader("ROC Curve")
        if len(set(y_test)) < 2:
            st.warning("測試資料僅有一個類別，無法繪製 ROC Curve。")
        else:
            positive_idx = list(model.classes_).index(1)
            y_prob = model.predict_proba(x_test)[:, positive_idx]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            ax.plot([0, 1], [0, 1], "k--", label="Random guess")
            ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
            ax.legend(loc="lower right")
            st.pyplot(fig)
            plt.close(fig)


def main():
    st.title("Predicting Diabetes Web App")
    st.markdown("Diabetes prediction and single-patient probability distribution")

    csv_path = Path(__file__).resolve().parent / "diabetes.csv"
    if not csv_path.is_file():
        st.error(f"找不到資料檔：{csv_path}。請將 diabetes.csv 放在 diabetes.py 同一資料夾。")
        st.stop()
    df = load_data(str(csv_path))
    if "Outcome" not in df.columns:
        st.error("資料檔缺少 Outcome 欄位。")
        st.stop()

    x = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    class_names = ["Non-Diabetes", "Diabetes"]

    st.sidebar.title("Model Selection Panel")
    st.sidebar.subheader("Dataset Split")
    train_percent = st.sidebar.slider("訓練資料比例 (%)", min_value=50, max_value=90, value=70, step=5)
    st.sidebar.caption(f"訓練集 {train_percent}% ／ 測試集 {100 - train_percent}%")

    st.sidebar.subheader("Select your Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Decision Tree", "Logistic Regression", "Random Forest"))

    st.sidebar.subheader("Model Parameters")
    if classifier == "Decision Tree":
        criterion = st.sidebar.radio("Criterion", ("gini", "entropy"))
        splitter = st.sidebar.radio("Splitter", ("best", "random"))
        parameters = (criterion, splitter)
    elif classifier == "Logistic Regression":
        c = st.sidebar.number_input("C (Regularization parameter)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        max_iter = st.sidebar.slider("Maximum number of iterations", min_value=100, max_value=500, value=300)
        parameters = (float(c), int(max_iter))
    else:
        n_estimators = st.sidebar.number_input("The number of trees in the forest", min_value=100, max_value=5000, value=100, step=10)
        max_depth = st.sidebar.number_input("Maximum depth", min_value=1, max_value=20, value=5, step=1)
        bootstrap = st.sidebar.radio("Bootstrap samples", (True, False))
        parameters = (int(n_estimators), int(max_depth), bool(bootstrap))

    metrics = st.sidebar.multiselect("Select your metrics", ("Confusion Matrix", "ROC Curve"), default=("Confusion Matrix", "ROC Curve"))

    # Stratify to keep both labels represented when possible.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=train_percent / 100, random_state=0, stratify=y
    )
    # Changing a widget does not retrain an unchanged model; results and predictions
    # share this cached model. New model/parameters/split requires one initial fit.
    model = get_trained_model(classifier, parameters, x_train, y_train)

    st.subheader(f"{classifier} Results")
    st.caption(f"Training: {len(x_train)} rows ({train_percent}%) · Testing: {len(x_test)} rows ({100 - train_percent}%)")
    y_pred = model.predict(x_test)
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    cols[1].metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
    cols[2].metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
    cols[3].metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")
    plot_metrics(metrics, model, x_test, y_test, class_names)

    st.divider()
    st.subheader("單筆預測 (Single Prediction)")
    st.write("輸入一位個案的特徵資料，使用目前選擇的已快取模型預測。")
    with st.form("single_patient_form"):
        values = {}
        input_cols = st.columns(2)
        for i, feature in enumerate(x.columns):
            series = pd.to_numeric(x[feature], errors="coerce")
            default = float(series.median()) if series.notna().any() else 0.0
            # Feature values are numeric as in the original diabetes dataset.
            with input_cols[i % 2]:
                values[feature] = st.number_input(
                    feature, value=round(default, 2), step=1.0, format="%.2f", key=f"feature_{feature}"
                )
        predict_clicked = st.form_submit_button("Predict / 預測")

    if predict_clicked:
        patient = pd.DataFrame([values], columns=x.columns)
        prediction = int(model.predict(patient)[0])
        probabilities = model.predict_proba(patient)[0]
        prob_by_class = dict(zip(model.classes_, probabilities))
        non_diabetes_prob = float(prob_by_class.get(0, 0.0))
        diabetes_prob = float(prob_by_class.get(1, 0.0))

        st.success(f"預測結果：{class_names[prediction]} ({'有糖尿病' if prediction == 1 else '無糖尿病'})")
        st.write(f"Non-Diabetes 機率：{non_diabetes_prob:.2%}")
        st.write(f"Diabetes 機率：{diabetes_prob:.2%}")
        fig, ax = plt.subplots()
        ax.pie(
            [non_diabetes_prob, diabetes_prob],
            labels=class_names,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#5DADE2", "#F1948A"],
        )
        ax.set_title("Single Prediction — Probability Distribution")
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("此機率為模型估計值，不能取代專業醫療診斷。")

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Diabetes Raw Dataset")
        st.dataframe(df)


if __name__ == "__main__":
    main()
