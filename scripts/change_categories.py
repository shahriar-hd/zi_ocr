import csv
import random
import sys
from pathlib import Path

# لیست ۵۰ دسته‌بندی (همان موارد بالا)
CATEGORIES = [
    "Holding", "Food Industry", "Steel & Metal", "Pharmaceutical", 
    "Construction & Real Estate", "Telecommunications", "Automotive", 
    "Petrochemical", "Insurance", "Hospitality & Tourism", "IT & Software", 
    "Banking & Finance", "Agriculture & Agro-industry", "Energy & Utilities", 
    "Defense & Aerospace", "Logistics & Transportation", "Mining", 
    "Water & Environment", "Oil & Gas Refining", "Pulp & Paper", 
    "Textile & Apparel", "Home Appliances", "Plastics & Rubber", 
    "Trading & Commerce", "Management Consulting", "Civil Engineering", 
    "Chemical Industry", "Cement & Building Materials", "Sugar & Sweeteners", 
    "Wood & Furniture", "Aviation", "Nanotechnology", "Maritime & Shipping", 
    "Printing & Packaging", "Medical Equipment", "Internet Service Provider", 
    "Dairy Products", "Oilfield Services", "Agrochemicals", 
    "Security & Defense", "Travel Agency", "Battery & Energy Storage", 
    "Alloys & Metallurgy", "Biotechnology", "Fintech", "Ceramics & Tiles", 
    "Waste Management", "Cosmetics & Personal Care", "Aerospace", 
    "Venture Capital"
]


def replace_categories_in_csv(input_file, output_file=None, category_column_name="category"):
    """
    جایگزینی تصادفی دسته‌بندی‌ها در یک فایل CSV
    
    Args:
        input_file (str): مسیر فایل ورودی CSV
        output_file (str): مسیر فایل خروجی (اختیاری - اگر None باشد، فایل ورودی بازنویسی می‌شود)
        category_column_name (str): نام ستون حاوی دسته‌بندی‌ها
    """
    
    # اگر فایل خروجی مشخص نشده، از فایل ورودی با پسوند _modified استفاده کن
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.stem + "_modified" + input_path.suffix
    
    # خواندن داده‌ها از فایل CSV
    rows = []
    headers = []
    
    try:
        with open(input_file, 'r', encoding='utf-8-sig') as file:
            reader = csv.reader(file)
            headers = next(reader)  # خواندن سطر هدر
            rows = list(reader)  # خواندن بقیه سطرها
            
    except FileNotFoundError:
        print(f"❌ خطا: فایل '{input_file}' یافت نشد!")
        return False
    except Exception as e:
        print(f"❌ خطا در خواندن فایل: {e}")
        return False
    
    # پیدا کردن اندیس ستون دسته‌بندی
    if category_column_name not in headers:
        print(f"❌ خطا: ستون '{category_column_name}' در فایل وجود ندارد!")
        print(f"   ستون‌های موجود: {', '.join(headers)}")
        return False
    
    category_index = headers.index(category_column_name)
    
    # بررسی اینکه آیا ستون دسته‌بندی در همه سطرها وجود دارد
    max_cols = max(len(row) for row in rows) if rows else 0
    if category_index >= max_cols:
        print(f"❌ خطا: ستون '{category_column_name}' در برخی سطرها خالی است!")
        return False
    
    # جایگزینی تصادفی دسته‌بندی‌ها
    print(f"🔄 در حال جایگزینی دسته‌بندی‌ها در {len(rows)} سطر...")
    
    for i, row in enumerate(rows):
        # اگر سطر کوتاه‌تر از اندیس ستون بود، آن را پر کن
        while len(row) <= category_index:
            row.append("")
        
        # جایگزینی با یک دسته‌بندی تصادفی
        random_category = random.choice(CATEGORIES)
        row[category_index] = random_category
    
    # نوشتن فایل خروجی
    try:
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)
            
        print(f"✅ فایل با موفقیت ذخیره شد: {output_file}")
        print(f"📊 تعداد سطرهای تغییر یافته: {len(rows)}")
        return True
        
    except Exception as e:
        print(f"❌ خطا در نوشتن فایل: {e}")
        return False


def main():
    """تابع اصلی برای اجرای اسکریپت از خط فرمان"""
    
    # تنظیمات پیش‌فرض
    input_file = "data/thesvg/logos.csv"
    output_file = "data/thesvg/logos.csv"
    column_name = "categories"
    
    # پردازش آرگومان‌های خط فرمان
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        column_name = sys.argv[3]
    
    print("=" * 50)
    print("🔄 جایگزین‌کننده تصادفی دسته‌بندی‌های CSV")
    print("=" * 50)
    print(f"📁 فایل ورودی: {input_file}")
    print(f"📁 فایل خروجی: {output_file if output_file else '(همان فایل با پسوند _modified)'}")
    print(f"📋 نام ستون: {column_name}")
    print(f"📊 تعداد دسته‌بندی‌های موجود: {len(CATEGORIES)}")
    print("-" * 50)
    
    # اجرای تابع اصلی
    success = replace_categories_in_csv(input_file, output_file, column_name)
    
    if success:
        print("\n✨ عملیات با موفقیت انجام شد!")
    else:
        print("\n❌ عملیات ناموفق بود!")
        sys.exit(1)


if __name__ == "__main__":
    main()