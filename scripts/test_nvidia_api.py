import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# ۱. بارگذاری کلید API از فایل .env
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError("لطفا NVIDIA_API_KEY را در فایل .env تنظیم کنید.")

# ۲. اتصال به API انویدیا (NVIDIA NIM)
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# ۳. مدل مدنظر (می‌توانید مدل‌های دیگر مثل meta/llama-3.1-70b-instruct را هم امتحان کنید)
MODEL_NAME = "qwen/qwen2.5-72b-instruct" 

# ۴. پرامپت جامع برای تولید نامه اداری مفصل
prompt_template = """
تو یک مشاور ارشد حقوقی و مدیر دفتر اداری مسلط به نگارش اسناد رسمی، صورت‌جلسه‌ها و نامه‌های پیچیده اداری در ایران هستی.

وظیفه تو نگارش یک نامه/درخواست اداری کاملاً جامع، مفصل، رسمی و با جزئیات کامل حقوقی است.

مشخصات نامه:
- موضوع نامه: {subject}
- نام شرکت/سازمان مبدأ: {company_name}
- حوزه فعالیت شرکت/سازمان مبدا: {company_category}
- نام سازمان/اداره مقصد: {destination_org}
- شرح خلاصه درخواست/نیاز: {summary_request}

قوانین و الزامات مهم نگارش:
۱. تمام داده‌های عددی و شناختی که در ورودی داده نشده‌اند (از جمله: شماره ثبت، شناسه ملی، شماره نامه، تاریخ، کد پستی، مبلغ، نام اعضا یا قطعات) را به‌صورت **رندوم اما کاملاً دقیق، واقعی‌نما و متناسب با عرف اداری ایران** تولید کن.
۲. بخش "body" باید **حداقل ۳ الی ۴ پاراگراف مفصل و کامل** باشد و شامل موارد زیر گردد:
   - مقدمه رسمی و اشاره به قوانین یا مصوبات مربوطه.
   - شرح کامل جزئیات، مشخصات فنی/حقوقی، دلایل و مستندات درخواست.
   - جداسازی شروط، تعهدات یا اقلام در قالب بندها یا شماره‌گذاری‌های شفاف داخل متن.
   - جمع‌بندی رسمی، قدردانی و تعیین مهلت یا درخواست اقدام عاجل.
۳. خروجی باید **صرفاً و فقط یک JSON معتبر** باشد. هیچ متن، سلام، مقدمه، توضیح یا علامتی مثل ```json قبل یا بعد از آن قرار نده.

فرمت دقیق خروجی JSON:
{{
  "company_name": "{company_name}",
  "destination_organization": "{destination_org}",
  "letter_number": "شماره نامه رندوم رسمی (مثلا: ۱۲۳/الف/۰۳)",
  "date": "تاریخ رندوم شمسی",
  "national_id": "شناسه ملی ۱۱ رقمی رندوم",
  "registration_number": "شماره ثبت رندوم ۵ رقمی",
  "subject": "{subject}",
  "body": "متن بسیار کامل، مفصل و حقوقی نامه شامل تمام پاراگراف‌ها، جزئیات، مبالغ، شروط و بندهای اداری...",
  "attachments": "لیست پیوست‌های فرضی",
  "signatory_title": "سمت امضاکننده",
  "contact_info": {{
    "address": "آدرس کامل اداری رندوم با کدپستی ۱۰ رقمی",
    "phone": "شماره تلفن ثابث رندوم همراه با کد استان",
    "email": "ایمیل رسمی رندوم",
    "website": "آدرس وب‌سایت رندوم"
  }}
}}
"""

# مقادیر نمونه برای تست
formatted_prompt = prompt_template.format(
    subject="درخواست تجهیزات بهمراه لیست فاکتور خرید",
    company_name="Dart",
    company_category="Language|Software",
    destination_org="نام شرکت مبدا رندوم",
    summary_request="شرح رندوم",
)

print("در حال ارسال درخواست به API انویدیا...")

# ۵. ارسال درخواست به API
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": formatted_prompt}
    ],
    temperature=0.3, # دمای پایین برای رعایت دقیق ساختار JSON
    max_tokens=2048,  # اجازه تولید متن‌های بسیار طولانی
    top_p=0.7
)

# ۶. دریافت و پارس کردن خروجی
raw_output = response.choices[0].message.content

try:
    # تبدیل خروجی متنی به دیکشنری پایتون (JSON Parse)
    json_data = json.loads(raw_output)
    print("\n--- خروجی با موفقیت به‌صورت JSON پارس شد ---\n")
    print(json.dumps(json_data, ensure_ascii=False, indent=2))
except json.JSONDecodeError:
    print("\n--- خروجی متنی دریافت شد اما JSON معتبر نبود ---")
    print(raw_output)