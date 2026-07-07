"""
ocr_bot.py
----------
Playwright automation for Nepal OCR e-Services company registration.
Follows the exact process from the OCR New Company Registration manual.

Flow:
  Step 1: Open site — user creates account / logs in manually
  Step 2: Name Reservation — bot fills form, user submits
  Step 3: Wait for name approval — user confirms via chat
  Step 4: Company Registration Form — bot fills all fields from collected data
  Step 5: Bot stops before Submit — user reviews and submits manually

Usage:
    bot = OCRBot(headless=False)
    bot.run(data)
"""

import time
from playwright.sync_api import sync_playwright, Page, TimeoutError as PlaywrightTimeout


SITE_URL = "https://www.ocr.gov.np/CRO/"


class OCRBot:
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.page: Page = None

                                                                                

    def run(self, data: dict, callback=None):
        """
        callback(stage, message) — called at each step so the frontend
        chat can show progress messages to the user.
        """
        self._cb = callback or (lambda stage, msg: print(f"[{stage}] {msg}"))

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(viewport={"width": 1280, "height": 900})
            self.page = context.new_page()

            try:
                self._step1_open_site()
                self._step2_manual_login()
                self._step3_name_reservation(data)
                self._step4_wait_for_approval()
                self._step5_company_registration(data)
                self._step6_done()
            finally:
                input("\n[OCR Bot] Press ENTER to close browser...")
                browser.close()

                                                                                

    def _step1_open_site(self):
        self._cb("open", "Opening OCR e-Services...")
        self.page.goto(SITE_URL, timeout=60000)
        self.page.wait_for_load_state("networkidle")
        self._cb("open", "Site loaded at https://www.ocr.gov.np/CRO/")

                                                                                

    def _step2_manual_login(self):
        self._cb("login", (
            "Please do the following manually in the browser:\n"
            "1. If you are a new user: click 'Create Company User Account'\n"
            "2. Fill your Full Name, Email, Phone, Gender, Citizenship No.\n"
            "3. Enter the CAPTCHA code and click 'Create User'\n"
            "4. Check your email for login credentials\n"
            "5. Log in with your Username and Password\n"
            "6. If first login: change your temporary password as required\n\n"
            "Once you are logged in and see the homepage, press ENTER here."
        ))
        input("\n>>> Press ENTER after you are logged in: ")
        self._cb("login", "Login confirmed. Proceeding to name reservation...")

                                                                                

    def _step3_name_reservation(self, data: dict):
        self._cb("name_reservation", "Navigating to Name Reservation form...")

                                           
        try:
            self.page.click("text=Name Check/Reservation", timeout=10000)
            self.page.wait_for_load_state("networkidle")
            self.page.click("text=Name Reservation Request Form", timeout=10000)
            self.page.wait_for_load_state("networkidle")
        except PlaywrightTimeout:
            self._cb("name_reservation", "Could not find menu automatically. Please navigate to Name Check/Reservation > Name Reservation Request Form manually.")
            input(">>> Press ENTER when you are on the Name Reservation Request Form: ")

                                                                                

                               
        company_type = data.get("company_type", "Private")
        try:
            self.page.select_option("select", label=company_type, timeout=5000)
        except:
            try:
                                                                   
                type_map = {
                    "Private": "प्राइभेट",
                    "Public": "पब्लिक",
                    "Partnership": "साझेदारी",
                }
                nepali = type_map.get(company_type, company_type)
                self.page.select_option("select", label=nepali)
            except:
                self._cb("name_reservation", f"⚠ Could not set company type to '{company_type}' — please set it manually.")

                                
        company_name_en = data.get("company_name_english", data.get("company_name", ""))
        self._safe_fill("input[name*='CompanyNameEnglish'], input[placeholder*='English']", company_name_en)

                                                              
        company_name_np = data.get("company_name_nepali", "")
        if company_name_np:
            self._safe_fill("input[name*='CompanyNameNepali'], input[placeholder*='Nepali']", company_name_np)

                                                                                
                                                         
        objectives = data.get("objectives", [])
        if isinstance(objectives, str):
            objectives = [{"nsic_code": "", "description": objectives}]

        for i, obj in enumerate(objectives):
            if i > 0:
                                                    
                try:
                    self.page.click("text=Add Objective", timeout=3000)
                    time.sleep(0.5)
                except:
                    pass

                                        
            nsic = obj.get("nsic_code", "") if isinstance(obj, dict) else ""
            desc = obj.get("description", obj) if isinstance(obj, dict) else str(obj)

            if nsic:
                try:
                    nsic_inputs = self.page.query_selector_all("input[name*='NsicCode'], td input[type='text']:first-child")
                    if i < len(nsic_inputs):
                        nsic_inputs[i].fill(str(nsic))
                        time.sleep(0.3)
                except:
                    pass

            if desc:
                try:
                    obj_inputs = self.page.query_selector_all("input[name*='Objective'], td input[type='text']:last-child")
                    if i < len(obj_inputs):
                        obj_inputs[i].fill(str(desc))
                except:
                    pass

        self._cb("name_reservation", (
            "✓ Name Reservation form filled.\n\n"
            f"  Company Name (EN): {company_name_en}\n"
            f"  Company Type: {company_type}\n"
            f"  Objectives: {len(objectives)} entered\n\n"
            "Please review the form, then click 'Submit' to submit the name reservation.\n"
            "You will receive a confirmation email. Once your name is APPROVED, press ENTER."
        ))
        input("\n>>> Press ENTER after your company name has been APPROVED by OCR: ")

                                                                                

    def _step4_wait_for_approval(self):
        self._cb("approval", "Good. Navigating to the company registration form...")

        try:
                                                                                      
            self.page.goto(SITE_URL, timeout=30000)
            self.page.wait_for_load_state("networkidle")
            time.sleep(2)

                                              
            self.page.click("text=Registration Form", timeout=10000)
            self.page.wait_for_load_state("networkidle")
        except PlaywrightTimeout:
            self._cb("approval", "Could not find Registration Form link automatically. Please navigate to it manually.")
            input(">>> Press ENTER when you are on the Company Registration Form: ")

                                                                                

    def _step5_company_registration(self, data: dict):
        self._cb("registration", "Filling Company Registration Form...")
        time.sleep(2)

                                                                                
                                                                 

                   
        self._safe_fill(
            "input[name*='Telephone'], input[placeholder*='Telephone']",
            data.get("contact_phone", "")
        )

                        
        self._safe_fill(
            "input[name*='Fax'], input[placeholder*='Fax']",
            data.get("fax_no", "")
        )

                       
        self._safe_fill(
            "input[name*='Email'], input[type='email']",
            data.get("contact_email", "")
        )

                            
        district = data.get("district", "")
        if district:
            try:
                self.page.select_option("select[name*='District']", label=district, timeout=3000)
            except:
                self._cb("registration", f"⚠ Could not set district '{district}' — please set manually.")

                          
        vdc = data.get("vdc_municipality", "")
        if vdc:
            try:
                self.page.select_option("select[name*='Muncipality'], select[name*='VDC']", label=vdc, timeout=3000)
            except:
                pass

        self._safe_fill("input[name*='WardNo'], input[placeholder*='Ward']",  data.get("ward_no", ""))
        self._safe_fill("input[name*='Street'], input[placeholder*='Street']", data.get("street", ""))
        self._safe_fill("input[name*='BlockNo'], input[placeholder*='Block']", data.get("block_no", ""))

        self._cb("registration", "✓ Company details filled.")

                                                                                
                                                         
        try:
            capital_type = data.get("capital_type", data.get("company_type_capital", ""))
            if capital_type:
                self.page.select_option(
                    "select[name*='CompanyType'], #CompanyType",
                    label=capital_type,
                    timeout=5000,
                )
                time.sleep(1)                                      
        except:
            self._cb("registration", "⚠ Could not set capital company type — please set manually.")

                                  
        self._safe_fill_by_label("Authorized Capital",  data.get("authorized_capital", ""))
        self._safe_fill_by_label("Authorized Rate",     data.get("authorized_rate", "100"))
        self._safe_fill_by_label("Quantity Of Shares",  data.get("quantity_of_shares", ""))
        self._safe_fill_by_label("Issued Capital",      data.get("issued_capital", ""))
        self._safe_fill_by_label("Paid Up Capital",     data.get("paid_up_capital", ""))

        self._cb("registration", "✓ Capital structure filled.")

                                                                                
        shareholders = data.get("shareholders", data.get("directors", []))
        for i, sh in enumerate(shareholders):
            self._add_shareholder(sh, index=i)

        self._cb("registration", f"✓ {len(shareholders)} shareholder(s) added.")

                                                                                
        doc_files = data.get("document_files", {})
                                                                
        if doc_files:
            self._upload_documents(doc_files)
        else:
            self._cb("registration", (
                "No documents were uploaded. If you have PDF scans ready, "
                "please upload them in the Document Details section manually.\n"
                "Required documents: Memorandum of Association, Articles of Association, etc."
            ))

        self._cb("registration", (
            "\n✅ Form filling complete!\n\n"
            "Please review ALL fields carefully in the browser.\n"
            "Click 'Preview' to check everything, then 'Submit' when you are ready.\n\n"
            "DO NOT submit yet — review first."
        ))

                                                                               

    def _step6_done(self):
        self._cb("done", (
            "The bot has finished filling the form.\n\n"
            "Next steps:\n"
            "1. Review the form carefully\n"
            "2. Click 'Preview' to see the full summary\n"
            "3. Click 'Submit' to submit your application\n"
            "4. After submission: visit OCR office with original documents\n\n"
            "You will receive a confirmation email after submission."
        ))

                                                                                

    def _add_shareholder(self, sh: dict, index: int):
        try:
                                                   
            self.page.click("text=Add Company Share Holder", timeout=5000)
            time.sleep(1)

                                                   
            sh_type = sh.get("type", "Person")
            try:
                self.page.select_option(
                    "select[name*='ShareHolderType'], .modal select, dialog select",
                    label=sh_type,
                    timeout=3000,
                )
                time.sleep(0.5)
            except:
                pass

            if sh_type == "Person":
                             
                self._safe_fill("input[name*='FirstName'][lang='en'], input[placeholder*='First'][lang='en']", sh.get("first_name", ""))
                self._safe_fill("input[name*='MiddleName'][lang='en']",                                        sh.get("middle_name", ""))
                self._safe_fill("input[name*='LastName'][lang='en'], input[placeholder*='Last'][lang='en']",   sh.get("last_name", sh.get("name", "")))

                                         
                self._safe_fill("input[name*='FirstName'][lang='ne']", sh.get("first_name_nepali", ""))
                self._safe_fill("input[name*='LastName'][lang='ne']",  sh.get("last_name_nepali", ""))

                        
                gender = sh.get("gender", "Male")
                try:
                    self.page.select_option("select[name*='Gender']", label=gender, timeout=2000)
                except:
                    pass

                                
                self._safe_fill("input[name*='FatherHusband']", sh.get("father_husband", ""))

                           
                is_foreigner = sh.get("foreigner", False)
                try:
                    radio_val = "Yes" if is_foreigner else "No"
                    self.page.check(f"input[type='radio'][value='{radio_val}']")
                except:
                    pass

                if is_foreigner:
                    self._safe_fill("input[name*='Passport']", sh.get("passport_no", ""))
                else:
                    self._safe_fill("input[name*='Citizenship']", sh.get("citizenship_no", ""))
                    district = sh.get("citizenship_district", "")
                    if district:
                        try:
                            self.page.select_option("select[name*='District']", label=district, timeout=2000)
                        except:
                            pass

                self._safe_fill("input[name*='Pan']", sh.get("pan_number", ""))

            elif sh_type == "Company":
                self._safe_fill("input[name*='CompanyEnglishName']",    sh.get("company_english_name", ""))
                self._safe_fill("input[name*='CompanyNepaliName']",     sh.get("company_nepali_name", ""))
                self._safe_fill("input[name*='CompanyRegistrationNo']", sh.get("company_reg_no", ""))
                self._safe_fill("input[name*='CompanyPanNo']",          sh.get("company_pan", ""))

                                                                                        
            roles = sh.get("roles", ["Founder", "Shareowner"])
            for role in roles:
                try:
                    self.page.check(f"input[type='checkbox'][value*='{role.upper()}'], label:has-text('{role}') input")
                except:
                    pass

                           
            self._safe_fill("input[name*='NoOfShares']",  str(sh.get("no_of_shares", "")))
            self._safe_fill("input[name*='TotalAmount']", str(sh.get("total_amount", "")))

                       
            witnesses = sh.get("witnesses", [])
            if witnesses:
                try:
                    self.page.fill("input[name*='NoOfWitness']", str(len(witnesses)))
                    time.sleep(0.5)
                    for j, w in enumerate(witnesses):
                        self._safe_fill(f"input[name*='WitnessFullName'][data-index='{j}']", w.get("full_name", ""))
                        self._safe_fill(f"input[name*='WitnessCitizenship'][data-index='{j}']", w.get("citizenship_no", ""))
                except:
                    pass

                                  
            try:
                self.page.click("text=Save", timeout=5000)
                time.sleep(1)
                self._cb("registration", f"  ✓ Shareholder {index + 1} saved")
            except:
                self._cb("registration", f"  ⚠ Could not auto-save shareholder {index + 1} — please click Save manually.")
                input(f">>> Press ENTER after saving shareholder {index + 1}: ")

        except Exception as e:
            self._cb("registration", f"  ⚠ Could not add shareholder {index + 1}: {e}\nPlease add manually.")

                                                                                

    def _upload_documents(self, doc_files: dict):
        """Upload PDF files to Document Details section."""
        self._cb("registration", "Uploading documents...")
        for doc_name, file_path in doc_files.items():
            if not file_path:
                continue
            try:
                                                                    
                row = self.page.locator(f"tr:has-text('{doc_name}')").first
                file_input = row.locator("input[type='file']")
                file_input.set_input_files(file_path)
                time.sleep(1)
                self._cb("registration", f"  ✓ Uploaded: {doc_name}")
            except Exception as e:
                self._cb("registration", f"  ⚠ Could not upload '{doc_name}': {e}\nPlease upload manually.")

                                                                                

    def _safe_fill(self, selector: str, value: str):
        """Try multiple comma-separated selectors, fill the first that works."""
        if not value:
            return
        for sel in selector.split(","):
            sel = sel.strip()
            try:
                self.page.fill(sel, str(value), timeout=3000)
                return
            except:
                continue

    def _safe_fill_by_label(self, label: str, value: str):
        if not value:
            return
        try:
            self.page.get_by_label(label, exact=False).fill(str(value), timeout=3000)
        except:
            pass