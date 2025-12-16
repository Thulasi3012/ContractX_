# app/services/database_service.py
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import json
from app.database.models import Document
from app.database.database import SessionLocal


class DatabaseService:
    """Service for handling database operations for document storage"""
    
    def __init__(self):
        self.db: Optional[Session] = None
    
    def get_db(self):
        """Get database session"""
        if not self.db:
            self.db = SessionLocal()
        return self.db
    
    def close_db(self):
        """Close database session"""
        if self.db:
            self.db.close()
            self.db = None
     
    def store_document(self, extraction_result: Dict[str, Any]) -> str:
        """
        Store extracted document data in the database
        """
        try:
            db = self.get_db()
            document_content = extraction_result.get("document_content")

            # Generate UUID
            document_id = str(uuid.uuid4())

            # Basic info
            filename = extraction_result.get("filename", "Unknown")
            total_pages = extraction_result.get("total_pages", 0)
            pages_data = extraction_result.get("pages", [])
            metadata = extraction_result.get("metadata", {})
            
            # Get or create overall summary
            overall_summary = extraction_result.get("overall_summary", {})
            if not overall_summary:
                print("[!] No overall_summary found, generating from page data...")
                overall_summary = self._generate_overall_summary(pages_data)
            
            entities = overall_summary.get("entities", {})

            # Top-level fields
            document_type = overall_summary.get("document_type")
            comprehensive_summary = overall_summary.get("summary", "")

            # Entity fields
            buyer_name = entities.get("buyer_name")
            seller_name = entities.get("seller_name")
            deadlines = entities.get("deadlines") or []
            alerts = entities.get("alerts") or []
            obligations = entities.get("obligations") or []
            dates = entities.get("dates") or []
            addresses = entities.get("addresses") or []
            contact_info = entities.get("contact_info") or {}
            other_parties = entities.get("other_parties") or []

            # Unified Parties JSON
            parties_json = {
                "buyer": buyer_name,
                "seller": seller_name,
                "other_parties": other_parties,
                "addresses": addresses,
                "contact_info": contact_info
            }

            # Text and JSON
            cleaned_text = self._compile_cleaned_text(pages_data)
            text_as_json = document_content

            print(f"\n[DB] Storing document:")
            print(f"  - Document Type: {document_type}")
            print(f"  - Buyer: {buyer_name}")
            print(f"  - Seller: {seller_name}")
            print(f"  - Deadlines: {len(deadlines)}")
            print(f"  - Alerts: {len(alerts)}")
            print(f"  - Obligations: {len(obligations)}")
            print(f"  - Summary length: {len(comprehensive_summary)} chars")

            # Insert DB Record
            document = Document(
                id=document_id,
                document_name=filename,
                uploaded_on=datetime.utcnow(),

                summary=comprehensive_summary,
                document_type=document_type,
                document_version=metadata.get("version", "4.0.0"),

                # Parties
                buyer=buyer_name,
                seller=seller_name,
                parties_json=parties_json,

                # Critical info
                deadlines=deadlines,
                alerts=alerts,
                obligations=obligations,

                # Text data
                cleaned_text=cleaned_text,
                text_as_json=text_as_json,

                # Metadata
                page_count=total_pages,
                extraction_method=metadata.get("extraction_method", "gemini-text-extractor")
            )

            # Save to DB
            db.add(document)
            db.commit()
            db.refresh(document)

            print(f"[OK] Document stored successfully with ID: {document_id}")
            return document_id

        except Exception as e:
            if db:
                db.rollback()
            print(f"[DB ERROR] Failed to store document: {str(e)}")
            raise

        finally:
            self.close_db()

    
    def _compile_cleaned_text(self, pages_data: list) -> str:
        """Compile cleaned text from all pages - FIXED"""
        text_parts = []
        
        for page in pages_data:
            page_num = page.get('page_number', 0)
            text_analysis = page.get('text_analysis', {})
            sections = text_analysis.get('sections', [])
            summary = text_analysis.get('summary', '')
            
            text_parts.append(f"\n{'='*60}")
            text_parts.append(f"PAGE {page_num}")
            text_parts.append(f"{'='*60}\n")
            
            # Add page summary
            if summary:
                text_parts.append(f"SUMMARY:\n{summary}\n")
            
            # Process sections recursively
            for section in sections:
                text_parts.extend(self._extract_section_text(section, level=0))
        
        return '\n'.join(text_parts)
    
    def _extract_section_text(self, section: Dict[str, Any], level: int = 0) -> list:
        """Recursively extract text from nested sections"""
        parts = []
        indent = "  " * level
        
        # Main heading
        heading = section.get('heading', '')
        if heading:
            parts.append(f"\n{indent}📌 {heading}")
        
        # Sub-headings
        sub_headings = section.get('sub_headings', [])
        for sub in sub_headings:
            sub_heading = sub.get('sub_heading', '')
            if sub_heading:
                parts.append(f"\n{indent}  └─ {sub_heading}")
            
            # Clauses
            clauses = sub.get('clauses', [])
            for clause in clauses:
                clause_text = clause.get('clause', '')
                if clause_text:
                    parts.append(f"{indent}    • {clause_text}")
                
                # Sub-clauses
                sub_clauses = clause.get('sub_clauses', [])
                for sub_clause in sub_clauses:
                    sub_clause_text = sub_clause.get('sub_clause', '')
                    if sub_clause_text:
                        parts.append(f"{indent}      ○ {sub_clause_text}")
        
        return parts
    
    def _generate_overall_summary(self, pages_data: list) -> Dict[str, Any]:
        """
        Generate comprehensive overall summary from all pages
        Combines individual page summaries into 4-5 paragraph overview
        """
        print("\n[SUMMARY] Generating overall document summary...")
        
        all_buyer_names = []
        all_seller_names = []
        all_dates = []
        all_deadlines = []
        all_alerts = []
        all_obligations = []
        all_addresses = []
        all_contact_info = {}
        page_summaries = []
        
        # Collect data from all pages
        for page in pages_data:
            text_analysis = page.get('text_analysis', {})
            entities = text_analysis.get('entities', {})
            summary = text_analysis.get('summary', '')
            
            # Collect entities
            if entities.get('buyer_name'):
                all_buyer_names.append(entities['buyer_name'])
            if entities.get('seller_name'):
                all_seller_names.append(entities['seller_name'])
            
            all_dates.extend(entities.get('dates', []))
            all_deadlines.extend(entities.get('deadlines', []))
            all_alerts.extend(entities.get('alerts', []))
            all_addresses.extend(entities.get('addresses', []))
            
            if entities.get('obligations'):
                all_obligations.extend(entities.get('obligations', []))
            
            if entities.get('contact_info'):
                all_contact_info.update(entities.get('contact_info', {}))
            
            if summary:
                page_summaries.append(summary)
        
        # Deduplicate
        buyer_name = all_buyer_names[0] if all_buyer_names else None
        seller_name = all_seller_names[0] if all_seller_names else None
        all_dates = list(set(filter(None, all_dates)))
        all_deadlines = list(set(filter(None, all_deadlines)))
        all_alerts = list(set(filter(None, all_alerts)))
        all_addresses = list(set(filter(None, all_addresses)))
        
        # Build comprehensive summary (4-5 paragraphs)
        summary_paragraphs = []
        
        # Paragraph 1: Document type and parties
        para1 = self._build_paragraph_1(pages_data, buyer_name, seller_name)
        if para1:
            summary_paragraphs.append(para1)
        
        # Paragraph 2: Key dates and deadlines
        para2 = self._build_paragraph_2(all_dates, all_deadlines)
        if para2:
            summary_paragraphs.append(para2)
        
        # Paragraph 3: Obligations and responsibilities
        para3 = self._build_paragraph_3(all_obligations, buyer_name, seller_name)
        if para3:
            summary_paragraphs.append(para3)
        
        # Paragraph 4: Critical alerts and special terms
        para4 = self._build_paragraph_4(all_alerts)
        if para4:
            summary_paragraphs.append(para4)
        
        # Paragraph 5: Contact and administrative information
        para5 = self._build_paragraph_5(all_addresses, all_contact_info)
        if para5:
            summary_paragraphs.append(para5)
        
        comprehensive_summary = "\n\n".join(summary_paragraphs)
        
        print(f"[OK] Overall summary generated ({len(comprehensive_summary)} chars)")
        
        return {
            "summary": comprehensive_summary,
            "document_type": self._detect_document_type(pages_data),
            "entities": {
                "buyer_name": buyer_name,
                "seller_name": seller_name,
                "dates": all_dates,
                "deadlines": all_deadlines,
                "alerts": all_alerts,
                "obligations": all_obligations,
                "addresses": all_addresses,
                "contact_info": all_contact_info
            }
        }
    
    def _build_paragraph_1(self, pages_data: list, buyer: str, seller: str) -> str:
        """Build paragraph 1: Document overview and parties"""
        parts = []
        
        doc_type = self._detect_document_type(pages_data)
        parts.append(f"This is a {doc_type}")
        
        if buyer and seller:
            parts.append(f" between {buyer} (Buyer/Client) and {seller} (Seller/Provider)")
        elif buyer:
            parts.append(f" involving {buyer} as the primary party")
        elif seller:
            parts.append(f" involving {seller} as the primary party")
        
        parts.append(f". The document consists of {len(pages_data)} pages and outlines the terms, conditions, and requirements for the engagement.")
        
        return ''.join(parts)
    
    def _build_paragraph_2(self, dates: list, deadlines: list) -> str:
        """Build paragraph 2: Key dates and deadlines"""
        if not dates and not deadlines:
            return ""
        
        parts = ["Key dates and deadlines include:"]
        
        if dates:
            date_str = ", ".join(dates[:5])
            parts.append(f" Important dates mentioned: {date_str}")
            if len(dates) > 5:
                parts.append(f" and {len(dates) - 5} more")
        
        if deadlines:
            deadline_str = "; ".join(deadlines[:3])
            parts.append(f". Critical deadlines: {deadline_str}")
            if len(deadlines) > 3:
                parts.append(f", and {len(deadlines) - 3} additional deadlines")
        
        parts.append(".")
        
        return ''.join(parts)
    
    def _build_paragraph_3(self, obligations: list, buyer: str, seller: str) -> str:
        """Build paragraph 3: Obligations and responsibilities"""
        if not obligations:
            return ""
        
        buyer_obligations = [o for o in obligations if o.get('party', '').lower() in ['buyer', 'client', 'purchaser']]
        seller_obligations = [o for o in obligations if o.get('party', '').lower() in ['seller', 'provider', 'contractor']]
        
        parts = ["Key obligations and responsibilities:"]
        
        if buyer_obligations and buyer:
            parts.append(f" {buyer} is responsible for {len(buyer_obligations)} primary obligation(s)")
        
        if seller_obligations and seller:
            if buyer_obligations:
                parts.append(f"; {seller} has {len(seller_obligations)} primary obligation(s)")
            else:
                parts.append(f" {seller} has {len(seller_obligations)} primary obligation(s)")
        
        if len(obligations) > 5:
            parts.append(f". Total of {len(obligations)} obligations outlined throughout the document")
        
        parts.append(".")
        
        return ''.join(parts)
    
    def _build_paragraph_4(self, alerts: list) -> str:
        """Build paragraph 4: Critical alerts and risks"""
        if not alerts:
            return ""
        
        parts = ["Important alerts and critical items:"]
        
        alerts_limited = alerts[:4]
        for i, alert in enumerate(alerts_limited, 1):
            parts.append(f" ({i}) {alert}")
            if i < len(alerts_limited):
                parts.append(";")
        
        if len(alerts) > 4:
            parts.append(f". Additionally, {len(alerts) - 4} more alerts are documented")
        
        parts.append(".")
        
        return ''.join(parts)
    
    def _build_paragraph_5(self, addresses: list, contact_info: dict) -> str:
        """Build paragraph 5: Contact and administrative information"""
        if not addresses and not contact_info:
            return ""
        
        parts = ["Contact and administrative information:"]
        
        if addresses:
            addr_str = "; ".join(addresses[:2])
            parts.append(f" Primary addresses: {addr_str}")
            if len(addresses) > 2:
                parts.append(f" and {len(addresses) - 2} additional address(es)")
        
        if contact_info:
            contact_items = []
            if contact_info.get('email'):
                contact_items.append(f"Email: {contact_info['email']}")
            if contact_info.get('phone'):
                contact_items.append(f"Phone: {contact_info['phone']}")
            if contact_info.get('website'):
                contact_items.append(f"Website: {contact_info['website']}")
            
            if contact_items:
                parts.append(f". Contact details: {', '.join(contact_items)}")
        
        parts.append(".")
        
        return ''.join(parts)
    
    def _detect_document_type(self, pages_data: list) -> str:
        """Detect document type from content"""
        type_keywords = {
            'Contract': ['contract', 'agreement'],
            'Purchase Agreement': ['purchase', 'sale', 'buyer', 'seller'],
            'Service Agreement': ['service', 'services'],
            'NDA': ['non-disclosure', 'confidential', 'nda'],
            'Lease': ['lease', 'rental', 'tenant'],
            'Employment Contract': ['employment', 'employee', 'position'],
            'Loan Agreement': ['loan', 'lender', 'borrower'],
            'Invoice': ['invoice', 'billing', 'payment'],
            'Purchase Order': ['purchase order', 'po', 'order'],
            'RFP': ['rfp', 'request for proposal', 'proposal']
        }
        
        all_text = ""
        for page in pages_data[:3]:
            text_analysis = page.get('text_analysis', {})
            all_text += text_analysis.get('summary', '') + " "
            all_text += str(text_analysis.get('entities', {})) + " "
        
        all_text_lower = all_text.lower()
        
        for doc_type, keywords in type_keywords.items():
            if any(keyword in all_text_lower for keyword in keywords):
                return doc_type
        
        return 'Contract'
    
    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """Retrieve document by ID"""
        try:
            db = self.get_db()
            document = db.query(Document).filter(Document.id == document_id).first()
            return document
        except Exception as e:
            print(f"[DB ERROR] Failed to retrieve document: {str(e)}")
            return None
        finally:
            self.close_db()
    
    def get_all_documents(self, skip: int = 0, limit: int = 100) -> list:
        """Retrieve all documents with pagination"""
        try:
            db = self.get_db()
            documents = db.query(Document).offset(skip).limit(limit).all()
            return documents
        except Exception as e:
            print(f"[DB ERROR] Failed to retrieve documents: {str(e)}")
            return []
        finally:
            self.close_db()
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document by ID"""
        try:
            db = self.get_db()
            document = db.query(Document).filter(Document.id == document_id).first()
            
            if document:
                db.delete(document)
                db.commit()
                print(f"[DB] Document {document_id} deleted successfully")
                return True
            else:
                print(f"[DB] Document {document_id} not found")
                return False
        except Exception as e:
            if db:
                db.rollback()
            print(f"[DB ERROR] Failed to delete document: {str(e)}")
            return False
        finally:
            self.close_db()