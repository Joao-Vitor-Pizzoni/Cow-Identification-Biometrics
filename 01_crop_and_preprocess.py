import os
import cv2

#BASE_PATH = "Vaquinhas"  # onde está seu dataset YOLO
OUTPUT_PATH = "dataset_final"

splits = ["train", "valid", "test"]

os.makedirs(OUTPUT_PATH, exist_ok=True)

def get_cow_id(filename):
    # exemplo: cow_12_001.jpg → 12
    parts = filename.split("_")
    return parts[0]

def process_split(split):
    images_path = os.path.join(split, "images")
    labels_path = os.path.join(split, "labels")
    output_split = os.path.join(OUTPUT_PATH, split)

    os.makedirs(output_split, exist_ok=True)

    for img_name in os.listdir(images_path):
        if not img_name.endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(images_path, img_name)
        label_path = os.path.join(labels_path, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        cow_id = get_cow_id(img_name)
        print(cow_id)
        '''
        class_folder = os.path.join(output_split, f"cow_{cow_id}")
        os.makedirs(class_folder, exist_ok=True)
        '''
        with open(label_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            cls, x, y, bw, bh = map(float, line.split())

            # YOLO → pixel
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)

            # evitar erro
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            filename = f"{img_name.split('_')[0]}.jpg"
            save_path = os.path.join(output_split, filename)

            cv2.imwrite(save_path, crop)

    print(f"[✔] {split} pronto")

for split in splits:
    process_split(split)

print("\n Dataset final para abstrair embeddings")
