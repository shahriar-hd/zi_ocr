#!/usr/bin/env python3
"""
استخراج پرامپت‌های کاربر و پاسخ‌های سیستمی از فایل JSON و تبدیل به Markdown
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


def is_user_message(item: Dict[str, Any]) -> bool:
    """تشخیص پیام کاربر"""
    return item.get('type') == 'user' and 'message' in item


def is_assistant_message(item: Dict[str, Any]) -> bool:
    """تشخیص پیام دستیار"""
    return item.get('type') == 'assistant' and 'message' in item


def extract_prompt_text(item: Dict[str, Any]) -> Optional[str]:
    """استخراج متن پرامپت کاربر"""
    if not is_user_message(item):
        return None
    
    message = item.get('message', {})
    content = message.get('content', [])
    
    for part in content:
        # گاهی پرامپت در tool_result یا بخش‌های دیگر است
        if isinstance(part, dict):
            if part.get('type') == 'text':
                return part.get('text', '')
            elif part.get('type') == 'tool_result':
                tool_content = part.get('content', '')
                if isinstance(tool_content, str):
                    return tool_content
                elif isinstance(tool_content, list):
                    for subpart in tool_content:
                        if isinstance(subpart, dict) and subpart.get('type') == 'text':
                            return subpart.get('text', '')
    
    return None


def extract_system_response_text(item: Dict[str, Any]) -> Optional[str]:
    """استخراج متن پاسخ سیستمی (فقط بخش text، بدون thinking و tool_use)"""
    if not is_assistant_message(item):
        return None
    
    message = item.get('message', {})
    content = message.get('content', [])
    
    text_parts = []
    
    for part in content:
        if isinstance(part, dict):
            # فقط بخش‌های type='text' را استخراج کن
            if part.get('type') == 'text':
                text_parts.append(part.get('text', ''))
    
    # اگه چندین text داشتیم، با خط جدید جدا کن
    return '\n\n'.join(text_parts) if text_parts else None


def get_timestamp(item: Dict[str, Any]) -> str:
    """استخراج زمان‌مهر با فرمت خوانا"""
    ts = item.get('timestamp', '')
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return ts
    return ''


def get_parent_uuid(item: Dict[str, Any]) -> str:
    """استخراج parentUuid برای ردیابی"""
    return item.get('parentUuid', '')


def process_file(file_path: str, output_dir: str = '.'):
    """پردازش فایل JSON و تولید Markdown"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # ترکیب خطوطی که ممکن است break شده باشند
    combined_content = ''.join(lines)
    
    # تقسیم بر اساس خطوط خالی جدید (هر JSON یک شیء جدا)
    # یا ساده‌تر: پارس کردن هر خط که ممکن است JSON کامل باشد
    items = []
    
    # روش پیشرفته‌تر: هر خط که با { شروع میشه و با } تموم میشه
    for line in lines:
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # ممکنه چند خطی باشه، اینجا هندل نمیشه
                pass
    
    # اگر خط‌ها جداگانه نبودند، سعی کن کل فایل رو پارس کنی
    if not items:
        try:
            data = json.loads(combined_content)
            if isinstance(data, list):
                items = data
            else:
                items = [data]
        except json.JSONDecodeError:
            # گاهی ساختار به صورت چندین JSON در یک فایل است
            # روش جایگزین: پیدا کردن JSON بین کرلی‌بریس‌ها
            import re
            json_pattern = r'\{[^{}]*\}'
            matches = re.findall(json_pattern, combined_content)
            for match in matches:
                try:
                    items.append(json.loads(match))
                except:
                    pass
    
    # گروه‌بندی بر اساس parentUuid و ساخت دیالوگ
    conversations = []
    current_conversation = []
    
    for item in items:
        parent = get_parent_uuid(item)
        if not parent:
            # شروع دیالوگ جدید
            if current_conversation:
                conversations.append(current_conversation)
            current_conversation = [item]
        else:
            current_conversation.append(item)
    
    if current_conversation:
        conversations.append(current_conversation)
    
    # نام فایل خروجی
    input_path = Path(file_path)
    output_name = f"{input_path.stem}_extracted.md"
    output_path = Path(output_dir) / output_name
    
    # تولید Markdown
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# استخراج پرامپت‌ها و پاسخ‌های سیستمی\n\n")
        f.write(f"**منبع:** `{file_path}`\n")
        f.write(f"**تاریخ استخراج:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        conv_index = 1
        for conv in conversations:
            f.write(f"## گفتگو #{conv_index}\n\n")
            
            # پیدا کردن parentUuid اولین عنصر به عنوان شناسه
            first_uuid = get_parent_uuid(conv[0]) if conv else ''
            if first_uuid:
                f.write(f"**شناسه گفتگو:** `{first_uuid}`\n\n")
            
            for item in conv:
                ts = get_timestamp(item)
                item_type = item.get('type', 'unknown')
                
                if ts:
                    f.write(f"[{ts}] ")
                
                if is_user_message(item):
                    # پرامپت کاربر
                    prompt_text = extract_prompt_text(item)
                    if prompt_text:
                        f.write(f"**👤 کاربر:**\n\n")
                        f.write(f"{prompt_text}\n\n")
                    else:
                        # ممکنه پرامپت در جای دیگه باشه
                        f.write(f"**👤 کاربر:** *(پرامپت غیر متنی)*\n\n")
                        
                elif is_assistant_message(item):
                    # پاسخ سیستمی
                    response_text = extract_system_response_text(item)
                    if response_text:
                        f.write(f"**🤖 سیستم:**\n\n")
                        f.write(f"{response_text}\n\n")
                    else:
                        # ممکنه فقط ابزار باشه یا think
                        content = item.get('message', {}).get('content', [])
                        has_tool = any(c.get('type') == 'tool_use' for c in content if isinstance(c, dict))
                        has_thinking = any(c.get('type') == 'thinking' for c in content if isinstance(c, dict))
                        
                        if has_tool:
                            f.write(f"**🤖 سیستم:** *(استفاده از ابزار)*\n\n")
                        elif has_thinking:
                            f.write(f"**🤖 سیستم:** *(در حال تفکر)*\n\n")
                        else:
                            f.write(f"**🤖 سیستم:** *(بدون متن قابل نمایش)*\n\n")
                
                elif item.get('type') == 'attachment':
                    # ممکنه پرامپت کاربر در attachment باشه
                    attach = item.get('attachment', {})
                    if attach.get('type') == 'queued_command':
                        prompt_list = attach.get('prompt', [])
                        for p in prompt_list:
                            if isinstance(p, dict) and p.get('type') == 'text':
                                f.write(f"**👤 کاربر (از attachment):**\n\n")
                                f.write(f"{p.get('text', '')}\n\n")
            
            f.write("---\n\n")
            conv_index += 1
    
    print(f"✅ فایل Markdown در '{output_path}' ذخیره شد.")
    return output_path


def main():
    if len(sys.argv) < 2:
        print("""
استفاده: python extract_conversation.py <فایل-json> [پوشه-خروجی]

مثال:
  python extract_conversation.py conversation.json
  python extract_conversation.py conversation.json ./output
        """)
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    
    if not os.path.exists(file_path):
        print(f"❌ خطا: فایل '{file_path}' یافت نشد.")
        sys.exit(1)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    process_file(file_path, output_dir)


if __name__ == "__main__":
    main()