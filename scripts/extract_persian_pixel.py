import os
import glob
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm

# ==================== تنظیمات ====================
INPUT_DIR = "/home/shahriar/Documents/zilber_ocr/data/hf/data/"              # مسیر پوشه‌ای که فایل‌های parquet در آن قرار دارند
OUTPUT_DIR = "/home/shahriar/Documents/zilber_ocr/data/hf/data/temp/"  # مسیر پوشه خروجی
# =================================================

def extract_parquet_files(input_dir, output_dir):
    # ساخت پوشه‌های خروجی
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # یافتن تمام فایل‌های parquet در دایرکتوری
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    
    if not parquet_files:
        print(f"هیچ فایل .parquet در مسیر '{input_dir}' یافت نشد.")
        return

    print(f"تعداد {len(parquet_files)} فایل parquet یافت شد.\n")

    all_metadata = []

    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        print(f"در حال پردازش فایل: {file_name}")

        try:
            # خواندن فایل parquet
            df = pd.read_parquet(file_path)
            
            # پیمایش تمام سطرها
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {file_name}"):
                row_dict = row.to_dict()
                extracted_row = {}

                # پردازش تک‌تک ستون‌ها و ویژگی‌ها
                for col_name, val in row_dict.items():
                    # اگر ستون مربوط به تصویر باشد
                    if col_name == 'image' or isinstance(val, (dict, bytes, Image.Image)):
                        try:
                            image = None
                            if isinstance(val, dict) and 'bytes' in val:
                                image = Image.open(io.BytesIO(val['bytes']))
                            elif isinstance(val, bytes):
                                image = Image.open(io.BytesIO(val))
                            elif isinstance(val, Image.Image):
                                image = val
                            
                            if image is not None:
                                # نام‌گذاری منحصربه‌فرد برای تصاویر
                                img_filename = f"{os.path.splitext(file_name)[0]}_row{idx}_{col_name}.png"
                                img_save_path = os.path.join(images_dir, img_filename)
                                image.save(img_save_path)
                                extracted_row[f"{col_name}_filename"] = img_filename
                            else:
                                extracted_row[col_name] = str(val)
                        except Exception as img_err:
                            extracted_row[col_name] = None
                    else:
                        # بقیه ویژگی‌ها و ستون‌ها
                        extracted_row[col_name] = val

                # افزودن نام فایل منبع برای پیگیری دقیق‌تر
                extracted_row['source_parquet'] = file_name
                all_metadata.append(extracted_row)

        except Exception as e:
            print(f"خطا در خواندن فایل {file_name}: {e}")

    # ذخیره متادیتا و تمام ویژگی‌های استخراج‌شده در یک فایل CSV
    if all_metadata:
        result_df = pd.DataFrame(all_metadata)
        csv_save_path = os.path.join(output_dir, "all_features_metadata.csv")
        result_df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ استخراج با موفقیت تمام شد!")
        print(f"📁 تصاویر در: {images_dir}")
        print(f"📊 تمام ویژگی‌ها و متن‌ها در: {csv_save_path}")

if __name__ == "__main__":
    extract_parquet_files(INPUT_DIR, OUTPUT_DIR)