
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Epilepsy Genetic Variant Predictor",
    page_icon="🧬"
)

st.title("🧬 Epilepsy Genetic Variant Analysis")

st.write(
    "Analyze genetic variants using a machine learning model "
    "and check their documented epilepsy association in ClinVar."
)

st.info(
    "Research tool only. Results are not a medical diagnosis "
    "or a clinical assessment of an individual."
)

st.markdown("---")


# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------
@st.cache_data
def load_dataset():
    return pd.read_excel("processed_epilepsy_variants.xlsx")


try:
    clinvar_data = load_dataset()

except FileNotFoundError:
    st.error(
        "Dataset file not found. Keep "
        "'cleaned_epilepsy_variants.xlsx' in the same folder "
        "as app.py."
    )
    st.stop()


# --------------------------------------------------
# 3. Input Features
# --------------------------------------------------
st.header("1. Enter Variant Details")

col1, col2 = st.columns(2)

with col1:

    gene = st.selectbox(
        "Gene",
        [
            "SCN1A",
            "KCNQ2",
            "DEPDC5",
            "PCDH19",
            "GABRG2",
            "Other"
        ]
    )

    variant_type = st.selectbox(
        "Variant Type",
        [
            "single nucleotide variant",
            "deletion",
            "insertion",
            "duplication",
            "indel"
        ]
    )

    genomic_pos = st.number_input(
        "Genomic Position (GRCh38)",
        min_value=0,
        value=0,
        step=1
    )

with col2:

    consequence = st.selectbox(
        "Molecular Consequence",
        [
            "missense",
            "nonsense",
            "splice",
            "frameshift",
            "synonymous",
            "other"
        ]
    )

    ref_allele = st.text_input(
        "Reference Allele",
        value="A"
    ).strip().upper()

    alt_allele = st.text_input(
        "Alternate Allele",
        value="C"
    ).strip().upper()


st.markdown("---")


# --------------------------------------------------
# 4. Analyze Variant
# --------------------------------------------------
if st.button("Analyze Variant", type="primary"):

    # Basic input validation
    if not ref_allele or not alt_allele:
        st.error("Please enter both reference and alternate alleles.")

    elif ref_allele == alt_allele:
        st.error("Reference and alternate alleles must be different.")

    else:

        # ==========================================
        # PART A: ML Pathogenicity Prediction
        # ==========================================
        st.header("2. ML Pathogenicity Prediction")

        try:
            model = joblib.load("epilepsy_variant_model.pkl")
            features = joblib.load("model_features.pkl")

            # Check that model has both classes
            if not hasattr(model, "classes_"):
                st.error("The model does not provide class information.")

            elif 0 not in model.classes_ or 1 not in model.classes_:
                st.warning(
                    "The trained model does not contain both class 0 "
                    "and class 1. A binary pathogenicity prediction "
                    "cannot be displayed."
                )

            elif not hasattr(model, "predict_proba"):
                st.error(
                    "The loaded model does not support probability "
                    "prediction."
                )

            else:

                # Create input row using training feature names
                input_df = pd.DataFrame(
                    0,
                    index=[0],
                    columns=features
                )

                # Numerical features
                if "ref_len" in input_df.columns:
                    input_df.loc[0, "ref_len"] = len(ref_allele)

                if "alt_len" in input_df.columns:
                    input_df.loc[0, "alt_len"] = len(alt_allele)

                if "indel_size" in input_df.columns:
                    input_df.loc[0, "indel_size"] = (
                        len(alt_allele) - len(ref_allele)
                    )

                if "genomic_position" in input_df.columns:
                    input_df.loc[0, "genomic_position"] = genomic_pos

                # Categorical features
                selected_features = [
                    f"Variant type_{variant_type}",
                    f"consequence_category_{consequence}",
                    f"Ref_Allele_{ref_allele}",
                    f"Alt_Allele_{alt_allele}",
                    f"Gene(s)_{gene}"
                ]

                for feature in selected_features:
                    if feature in input_df.columns:
                        input_df.loc[0, feature] = 1

                # Find the probability corresponding to class 1
                probabilities = model.predict_proba(input_df)[0]

                class_1_index = list(model.classes_).index(1)

                prob = probabilities[class_1_index]

                # Use the existing threshold for now.
                # It must be validated using your training data.
                threshold = 0.64

                if prob >= threshold:
                    st.warning(
                        "ML Research Classification: "
                        "**Pathogenic class (1)**"
                    )
                else:
                    st.success(
                        "ML Research Classification: "
                        "**Class 0**"
                    )

                st.metric(
                    "Model Score for Class 1",
                    f"{prob * 100:.2f}%"
                )

                st.progress(float(prob))

                st.caption(
                    "This is the model's score for class 1, "
                    "not a confirmed clinical probability."
                )

        except FileNotFoundError:
            st.error(
                "Model files not found. Make sure "
                "epilepsy_variant_model.pkl and "
                "model_features.pkl are in the app folder."
            )

        except Exception as e:
            st.error(f"ML prediction error: {e}")


        
        # ==========================================
        # PART B: Epilepsy Association
        # ==========================================
        st.markdown("---")
        st.header("3. Epilepsy Association")

        st.write(
            "Checking your dataset for a matching variant "
            "and documented epilepsy-related condition."
        )

        # Step 1: Match gene, reference and alternate alleles
        matched = clinvar_data[
            (
                clinvar_data["Gene(s)"]
                .fillna("")
                .str.upper()
                .str.contains(gene.upper(), regex=False)
            )
            &
            (
                clinvar_data["Ref_Allele"]
                .fillna("")
                .str.upper() == ref_allele
            )
            &
            (
                clinvar_data["Alt_Allele"]
                .fillna("")
                .str.upper() == alt_allele
            )
        ].copy()

        # Step 2: Match genomic position if provided
        if genomic_pos != 0 and not matched.empty:
            matched = matched[
                pd.to_numeric(
                    matched["Genomic_Pos"],
                    errors="coerce"
                ) == genomic_pos
            ]

        # Step 3: Check epilepsy-related conditions
        if matched.empty:

            st.error(
                "❓ EPILEPSY ASSOCIATION: NOT FOUND "
                "IN YOUR DATASET"
            )

            st.write(
                "No matching variant record was found "
                "using the entered gene, alleles, "
                "and genomic position."
            )

            st.info(
                "This does not mean the variant is benign "
                "or unrelated to epilepsy. It means your "
                "current dataset does not contain a matching record."
            )

        else:

            # Search the condition column for epilepsy-related terms
            condition_text = (
                matched["Condition(s)"]
                .fillna("")
                .astype(str)
                .str.lower()
            )

            epilepsy_terms = (
    r"epilepsy|epileptic|seizure|"
    r"dravet|developmental and epileptic|"
    r"\bdee\b|"
    r"scn1a-related disorder"
)

            epilepsy_records = matched[
                condition_text.str.contains(
                    epilepsy_terms,
                    regex=True,
                    na=False
                )
            ]

            # Step 4: Display association result
            if not epilepsy_records.empty:

                st.success(
                    "✅ EPILEPSY ASSOCIATION: FOUND "
                    "IN YOUR DATASET"
                )

                st.write(
                    "Your dataset contains a matching variant "
                    "with an epilepsy-related condition."
                )

                st.subheader("Documented ClinVar Evidence")

                for _, row in epilepsy_records.iterrows():

                    with st.container(border=True):

                        st.write(
                            "**Variation ID:**",
                            row.get("VariationID", "N/A")
                        )

                        st.write(
                            "**Gene:**",
                            row.get("Gene(s)", "N/A")
                        )

                        st.write(
                            "**Associated Condition:**",
                            row.get("Condition(s)", "N/A")
                        )

                        st.write(
                            "**Germline Classification:**",
                            row.get(
                                "Germline classification",
                                "N/A"
                            )
                        )

                        st.write(
                            "**Molecular Consequence:**",
                            row.get(
                                "Molecular consequence",
                                "N/A"
                            )
                        )

                        st.write(
                            "**Reference Allele:**",
                            row.get("Ref_Allele", "N/A")
                        )

                        st.write(
                            "**Alternate Allele:**",
                            row.get("Alt_Allele", "N/A")
                        )

                        st.write(
                            "**Genomic Position:**",
                            row.get("Genomic_Pos", "N/A")
                        )

                st.info(
                    "This indicates a documented epilepsy-related "
                    "condition in the matching dataset record. "
                    "It does not establish a diagnosis in a person."
                )

            else:

                st.warning(
                    "⚠️ EPILEPSY ASSOCIATION: "
                    "NOT DOCUMENTED IN MATCHING RECORDS"
                )

                st.write(
                    "A matching variant record was found, "
                    "but its Condition(s) field does not contain "
                    "the epilepsy-related terms checked by this app."
                )

                st.subheader("Matching ClinVar Records")

                for _, row in matched.iterrows():

                    st.write(
                        "**Condition:**",
                        row.get("Condition(s)", "N/A")
                    )

                    st.write(
                        "**Classification:**",
                        row.get(
                            "Germline classification",
                            "N/A"
                        )
                    )

                st.info(
                    "This is not proof that the variant has "
                    "no epilepsy association. The condition "
                    "may be recorded under another name, "
                    "or evidence may be absent from your dataset."
                )