import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import streamlit as st
import torch
import random
import os
import pickle
from PIL import Image
from glob import glob
from src.infer import ModelInferenceVisualizer 
from data.parse import CustomDataset

st.set_page_config(page_title="Image Classification Demo", layout="wide")
@st.cache_resource
def load_model(model_path, model_name, num_classes, device):
    import timm
    model = timm.create_model(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))    
    return model.eval().to(device)

@st.cache_data
def load_class_names(pkl_path):
    with open(pkl_path, "rb") as fp: return pickle.load(fp)

class StreamlitApp:
    def __init__(self, ds_nomi, model_name):
        self.ds_nomi = ds_nomi
        self.model_name = model_name        
        self.lang_code = "en" 

        self.LANGUAGES = {
            "en": {
                "title": "AI Model Inference Visualization",
                "description": "Select or upload an image to run inference and visualize the results.",
                "upload_button": "Upload Your Image",
                "random_images_label": "Select a Random Image",
                "result_label": "Model Results",
                "accuracy_label": "Model Accuracy:",
                "f1_score_label": "F1 Score:",
                "select_language": "Select Language",
            },
            "ko": {
                "title": "AI 모델 추론 시각화",
                "description": "이미지를 선택하거나 업로드하여 추론을 실행하고 결과를 시각화합니다.",
                "upload_button": "이미지 업로드",
                "random_images_label": "랜덤 이미지 선택",
                "result_label": "모델 결과",
                "accuracy_label": "모델 정확도:",
                "f1_score_label": "F1 점수:",
                "select_language": "언어 선택",
            }
        }

    def run(self):
        sample_ims_dir = "demo_ims"
        ims_dir = "/home/bekhzod/Desktop/backup/image_classification_project_datasets"

        language = st.selectbox(
            self.LANGUAGES['en']['select_language'],
            options=['English', 'Korean'],
            index=0
        )
        self.lang_code = "en" if language == "English" else "ko"

        st.title(self.LANGUAGES[self.lang_code]["title"])
        st.write(self.LANGUAGES[self.lang_code]["description"])

        # Load class names and model
        class_names = load_class_names(f"saved_cls_names/{self.ds_nomi}_cls_names.pkl")
        device = "cpu"
        model = load_model(
            model_path=f"saved_models/{self.ds_nomi}_best_model.pth",
            model_name=self.model_name,
            num_classes=len(class_names),
            device=device
        )

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        model_inference = ModelInferenceVisualizer(
            model=model,
            device=device,
            mean=mean,
            std=std,
            outputs_dir=None,
            ds_nomi=self.ds_nomi,
            class_names=class_names
        )

        # Prepare sample images
        save_dir = os.path.join(sample_ims_dir, self.ds_nomi)
        os.makedirs(save_dir, exist_ok=True)
        sample_image_paths = glob(os.path.join(save_dir, "*.png"))

        if len(sample_image_paths) == 0:
            ds = CustomDataset(ims_dir, data_type="val", ds_nomi=self.ds_nomi) if self.ds_nomi == "facial_expression" else CustomDataset(ims_dir, ds_nomi=self.ds_nomi)             
            random_images = random.sample(ds.im_paths, 5)
            for idx, path in enumerate(random_images):
                cls_name = os.path.basename(path).split("_")[0]  if self.ds_nomi in ["lentils", "apple_disease"] else os.path.basename(os.path.dirname(path))
                with Image.open(path).convert("RGB") as im:
                    im.save(os.path.join(save_dir, f"sample_im_{idx + 1}___{cls_name}.png"))
            sample_image_paths = glob(os.path.join(save_dir, "*.png"))

        # UI: Image selection or upload
        selected_image = st.selectbox(self.LANGUAGES[self.lang_code]["random_images_label"], sample_image_paths)
        uploaded_image = st.file_uploader(self.LANGUAGES[self.lang_code]["upload_button"], type=["jpg", "png", "jpeg"])

        im_path = uploaded_image if uploaded_image else selected_image

        if im_path:
            with st.spinner("Running inference..."):
                result = model_inference.demo(im_path)
                predicted_class = list(class_names.keys())[result['pred']]
                ground_truth_label = result['gt']

            st.subheader(self.LANGUAGES[self.lang_code]["result_label"])

            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)

            with row1_col1:
                st.markdown(f"<h3 style='text-align: center;'> Original Input Image </h3>", unsafe_allow_html=True)
                st.image(result["original_im"], use_container_width=True)

            with row1_col2:
                st.markdown(f"<h3 style='text-align: center;'> The Model Decision Heatmap </h3>", unsafe_allow_html=True)
                st.image(result["gradcam"], use_container_width=True)
                
            with row2_col1:
                st.markdown(f"<h3 style='text-align: center;'>GT: {ground_truth_label}</h3>", unsafe_allow_html=True)
                st.image(result["original_im"], use_container_width=True)

            with row2_col2:
                st.markdown(f"<h3 style='text-align: center; color: green;'>PRED: {predicted_class}</h3>", unsafe_allow_html=True)
                st.image(result["probs"], use_container_width=True)
                
                st.markdown(
                    f"""
                    <div style='text-align: center; width: 100%;'>
                        The model is {(result["confidence"]):.2f}% confident that the image belongs to → {predicted_class} class!
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.warning("Please select or upload an image.")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_nomi", type=str, required=True, help="Dataset name")
    parser.add_argument("--model_name", type=str, default="rexnet_150", help="Model architecture from timm")    
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Directory to store outputs")
    return parser.parse_args()

if __name__ == "__main__":
    available_datasets = [os.path.basename(pth).split("_best_model")[0] for pth in glob(f"saved_models/*.pth")]
    ds_nomi = st.sidebar.selectbox("Choose Dataset", options=available_datasets, index=0)
    model_name = st.sidebar.text_input("Model name", value="rexnet_150")    

    app = StreamlitApp(
        ds_nomi=ds_nomi,
        model_name=model_name        
    )
    app.run()