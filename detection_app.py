import streamlit as st
from PIL import Image
import os
import sys
from models import WrapperYOLO, MorphBasedDetector, SegmentBaseDetector
import torch


torch.classes.__path__ = []

IMAGES_PATH = os.path.join('.', 'eggs')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')


def detect(image, model_name):
    model = st.session_state.models[model_name]
    return model.detect(image)


def get_image_files(folder_path, level):
    names = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(IMAGE_EXTENSIONS)]
    if level == 'Begginer':
        return [name for name in names if int(name[:2]) < 20]
    if level == 'Itermediate':
        return [name for name in names if int(name[:2]) < 30 and int(name[:2]) >= 20]
    if level == 'Expert':
        return [name for name in names if int(name[:2]) >= 30]


def crop_center_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return img.crop((left, top, right, bottom))


def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'input'
        st.session_state.result_page_num = 0
        st.session_state.results = []
        st.session_state.selected_files = set()
        st.session_state.selected_images = []
        st.session_state.original_image = None
        st.session_state.models = {
            'YOLOv11': WrapperYOLO(os.path.join('.', 'yolo-best.pt'), conf=0.6),
            'YOLOv11-small': WrapperYOLO(os.path.join('.', 'yolo-small-best.pt'), conf=0.5),
            'MorphBasedDetector': MorphBasedDetector(),
            'SegmentBaseDetector': SegmentBaseDetector(),
        }

    if st.session_state.page == 'input':
        render_input_page()
    elif st.session_state.page == 'results':
        render_results_page()

def render_input_page():
    st.title("Приложение для обнаружения и подсчета яиц")
    st.subheader("Информация об алгоритмах:")
    st.markdown('\n'.join([
        '- SegmentBaseDetector - базовый алгоритм для сегментации',
        '- MorphBasedDetector - специальный алгоритм для шумного фона',
        '- YOLOv11 - сверточная нейронная сеть',
        '- YOLOv11-small - сверточная нейронная сеть, дообученная на всего 5 фотографиях',
    ]))

    st.sidebar.header("Settings")

    # Model selection
    selected_model = st.sidebar.selectbox(
        "Выбрать алгоритм:",
        ["SegmentBaseDetector", "MorphBasedDetector", "YOLOv11", 'YOLOv11-small'],
    )

    # Image source selection
    image_source = st.sidebar.radio(
        "Как загрузить изображение?",
        ["Выбрать из доступных", "Загрузить свое"]
    )

    image = None
    if image_source == "Выбрать из доступных":
        selected_level = st.sidebar.selectbox(
            "Выберите уровень",
            ["Begginer", "Itermediate", "Expert"],
        )

        # Title
        st.title("Выберите изображения для тестирования алгоритмов")

        # Load images
        st.session_state.level_files = []
        if st.session_state.level_files:
            image_files = st.session_state.level_files
        else:
            image_files = get_image_files(IMAGES_PATH, level=selected_level)
            st.session_state.level_files = image_files
        cols = st.columns(5)

        # Display images and clickable buttons
        for idx, image_file in enumerate(image_files):
            img_path = os.path.join(IMAGES_PATH, image_file)
            image = crop_center_square(Image.open(img_path))

            with cols[idx % 5]:  # Place image in column
                st.image(image, caption=image_file, use_container_width=True)
                if image_file not in st.session_state.selected_files:
                    if st.button(f"Выбор {image_file}", key=f"select_{idx}"):
                        st.session_state.selected_files.add(image_file)
                        st.rerun()
                else:
                    if st.button(f"Убрать {image_file}", key=f"deselect_{idx}"):
                        st.session_state.selected_files.remove(image_file)
                        st.rerun()

        st.markdown("### Выбранные изображения:")
        if st.session_state.selected_files:
            selected_list = ", ".join([f"{file}" for file in sorted(st.session_state.selected_files)])
            st.markdown(selected_list)
        else:
            st.write("Вы пока не выбрали яйца.")
    else:
        uploaded_file = st.sidebar.file_uploader("Загрузить яйца:", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
        if image:
            st.image(image, caption="Выбранное изображение", use_container_width=True)
            st.session_state.selected_images = [image]


    # Detection button
    if st.sidebar.button("Запустить детекцию яиц"):
        is_good = False
        if not selected_model:
            st.sidebar.error("Please select at least one model")
        if not st.session_state.selected_images:
            if not st.session_state.selected_files:
                st.sidebar.error("Please select/upload an image")
            else:
                st.session_state.selected_images = [Image.open(os.path.join(IMAGES_PATH, name)) for name in sorted(st.session_state.selected_files)]
                is_good = True
        else:
            is_good = True
        if is_good:
            # Process image with selected models
            results = []
            for image in st.session_state.selected_images:
                figs, white, red = detect(image, selected_model)
                results.append({
                    "model": selected_model,
                    "figs": figs,
                    "white": white,
                    "red": red
                })
            
            st.session_state.results = results
            st.session_state.page = 'results'
            st.rerun()

def render_results_page():
    st.title("Результаты детекции яиц")
    
    col1, col2, col3 = st.columns(3)

    with col2:
        if st.button("Назад на главную"):
            st.session_state.page = 'input'
            st.session_state.selected_images = []
            st.session_state.selected_files = set()
            st.session_state.result_page_num = 0
            st.rerun()

    with col1:
        if st.session_state.result_page_num > 0:
            if st.button("<- Предыдущий результат"):
                st.session_state.result_page_num -= 1
                st.rerun()
    
    with col3:
        if st.session_state.result_page_num < len(st.session_state.results) - 1:
            if st.button("Следующий результат ->"):
                st.session_state.result_page_num += 1
                st.rerun()

    idx = st.session_state.result_page_num
    result = st.session_state.results[idx]
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Начальное изображение")
        st.image(st.session_state.selected_images[idx], use_container_width=True)
    with col2:
        st.subheader("Результаты подсчета")
        st.metric("Найдено белых яиц", result['white'])
        st.metric("Найдено коричневых (red) яиц", result['red'])
    st.subheader(f"{result['model']} detection visualization")
    for fig in result['figs']:
        st.pyplot(fig)


if __name__ == "__main__":
    main()