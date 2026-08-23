from huggingface_hub import HfApi, login

# ۱. ورود به حساب (اگر قبلاً لاگین نکرده‌اید)
# توکن خود با دسترسی Write را وارد کنید
login()

api = HfApi()
repo_id = "shahriarhd/persian-official-letters-synth"
repo_type = "dataset"  # اگر ریپوزیتوری شما از نوع Dataset است (که در کد قبلی بود)

# ۲. دریافت لیست تمام فایل‌های موجود در ریپوزیتوری
files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

# ۳. فیلتر کردن فایل‌های سیستمی گیت (مثل .gitattributes) تا پاک نشوند
files_to_delete = [f for f in files if not f.startswith(".")]

print(f"تعداد {len(files_to_delete)} فایل پیدا شد:")
for f in files_to_delete:
    print(f"- {f}")

# ۴. حذف یکجای فایل‌ها
if files_to_delete:
    # ساخت لیستی از عملیات حذف
    from huggingface_hub import CommitOperationDelete

    operations = [
        CommitOperationDelete(path_in_repo=f) for f in files_to_delete
    ]

    api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        operations=operations,
        commit_message="Clear dataset repository for fresh run",
    )
    print("\n✅ تمامی فایل‌های قبلی با موفقیت پاک شدند.")
else:
    print("\nریپوزیتوری از قبل خالی است.")