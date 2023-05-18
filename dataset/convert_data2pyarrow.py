import os
import glob
import pyarrow as pa
import pandas as pd


def read_image(file_path: str):
    with open(file_path, 'rb') as f:
        img = f.read()
    return img


def save2arrow(input_dir, output_dir, split="train"):
    os.makedirs(output_dir, exist_ok=True)

    img_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    img_list = []

    col_names = ["image", "label"] if split == "train" else ["image", "img_id"]
    for file_path in img_files:
        img = read_image(file_path)
        if split == "train":
            label = file_path.split(os.sep)[-1].split(".")[0]
            img_list.append({"image": img, "label": label})
        else:
            img_id = file_path.split(os.sep)[-1].split(".")[0]
            img_list.append({"image": img, "img_id": int(img_id)})

    table = pa.Table.from_pandas(pd.DataFrame(
        img_list, columns=col_names))

    with pa.OSFile(os.path.join(output_dir, f"{split}.arrow"), "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


def main():

    for split in ["train", "test"]:
        input_dir = 'data'
        output_dir = 'data_parquet'
        input_dir = os.path.join(input_dir, split)
        output_dir = os.path.join(output_dir, split)
        save2arrow(input_dir, output_dir, split=split)


if __name__ == '__main__':
    main()
