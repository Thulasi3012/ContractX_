"""
LLM Usage Tracker
Tracks token usage, costs, and LLM calls for different LLM types (text, image, table, chat_agent)
Updates database with aggregated statistics per document
"""

import time
from typing import Dict, Any, Optional, Literal
from datetime import datetime
from sqlalchemy.orm import Session
from app.database.models import LLMUsage
from app.database.database import get_db
from uuid import UUID
from app.config.config import Config

# Gemini API Pricing (as of 2025) - Update these based on current pricing
GEMINI_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-2.0-flash-exp": {"input_per_million": 0.0, "output_per_million": 0.0},
    "gemini-2.0-flash": {"input_per_million": 0.25, "output_per_million": 1.0},  # example pricing
    "gemini-2.5-flash": {"input_per_million": 0.10, "output_per_million": 0.40},
    "gemini-1.5-flash": {"input_per_million": 0.075, "output_per_million": 0.30},
    "gemini-1.5-pro": {"input_per_million": 1.25, "output_per_million": 5.00},
}

# USD to INR conversion rate (update as needed)
USD_TO_INR = 83.0

LLMType = Literal["text", "image", "table", "chat_agent"]


class LLMUsageTracker:
    """
    Tracks LLM usage for a specific LLM type (text/image/table/chat_agent)
    """
    
    def __init__(
        self, 
        llm_type: LLMType,
        model_name: str = Config.GEMINI_MODEL,
        document_id: Optional[UUID] = None
    ):
        """
        Initialize tracker for specific LLM type
        
        Args:
            llm_type: Type of LLM (text, image, table, chat_agent)
            model_name: Gemini model being used
            document_id: UUID of document being processed (optional, can be set later)
        """
        self.llm_type = llm_type
        self.model_name = model_name
        self.document_id = document_id
        
        # Session tracking
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0
        self.cost_in_inr = 0.0
        
        # Request tracking
        self._current_request_start = None
        self._current_request_prompt = None
        
        # Labels for display
        self.labels = {
            "text": "Text LLM",
            "image": "Image LLM", 
            "table": "Table LLM",
            "chat_agent": "Chat LLM"
        }
        
        print(f"[LLM Tracker] Initialized {self.labels[llm_type]} tracker (model: {model_name})")
    
    def set_document_id(self, document_id: UUID):
        """Set document ID after tracker initialization"""
        self.document_id = document_id
    
    def start_request(self, prompt: str):
        """Mark the start of an LLM request"""
        self._current_request_start = time.time()
        self._current_request_prompt = prompt
        
        # Estimate input tokens (rough approximation: 1 token ≈ 4 chars)
        estimated_input = len(prompt) // 4
        self.input_tokens += estimated_input
    
    def end_request(self, response: Any):
        """
        Mark the end of an LLM request and extract token usage
        
        Args:
            response: Gemini API response object
        """
        if self._current_request_start is None:
            print(f"[WARNING] end_request called without start_request for {self.llm_type}")
            return
        
        elapsed = time.time() - self._current_request_start
        
        try:
            # Extract token usage from Gemini response
            usage_metadata = response.usage_metadata
            
            input_tok = usage_metadata.prompt_token_count
            output_tok = usage_metadata.candidates_token_count
            
            # Update cumulative stats
            self.input_tokens = input_tok  # Use actual count, not estimated
            self.output_tokens += output_tok
            self.llm_calls += 1
            
            # Calculate cost
            cost_usd = self._calculate_cost(input_tok, output_tok)
            cost_inr = cost_usd * USD_TO_INR
            self.cost_in_inr += cost_inr
            
            print(f"[{self.labels[self.llm_type]}] Call #{self.llm_calls}:")
            print(f"  Input: {input_tok} tokens | Output: {output_tok} tokens")
            print(f"  Cost: ₹{cost_inr:.2f} ({elapsed:.2f}s)")
            
        except AttributeError as e:
            # Fallback if response doesn't have usage_metadata
            print(f"[WARNING] Could not extract token usage: {e}")
            output_tok = len(response.text) // 4 if hasattr(response, 'text') else 0
            self.output_tokens += output_tok
            self.llm_calls += 1
        
        finally:
            self._current_request_start = None
            self._current_request_prompt = None
    
    #to get an current model 
    def get_current_model(self) -> str:
        return getattr(Config, "GEMINI_MODEL", "gemini-2.5-flash")

    def get_gemini_pricing(self, model_name: str) -> Dict[str, float]:
        if model_name not in GEMINI_PRICING:
            fallback = "gemini-2.5-flash"
            print(f"[WARNING] Unknown model '{model_name}', falling back to '{fallback}' pricing")
            return GEMINI_PRICING[fallback]
        return GEMINI_PRICING[model_name]

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str = None) -> float:
        if model_name is None:
            model_name = self.get_current_model()
        pricing = self.get_gemini_pricing(model_name)
        input_cost = (input_tokens / 1_000_000) * pricing["input_per_million"]
        output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]
        return input_cost + output_cost

        
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics for this tracker"""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_in_inr": round(self.cost_in_inr, 2),
            "llm_calls": self.llm_calls,
            "label": self.labels[self.llm_type]
        }
    
    def save_to_db(self, db: Session = None):
        """
        Save or update tracker stats in database
        
        Args:
            db: SQLAlchemy session (optional, will create new if not provided)
        """
        if not self.document_id:
            print(f"[WARNING] Cannot save to DB: document_id not set for {self.llm_type}")
            return
        
        close_db = False
        if db is None:
            db = next(get_db())
            close_db = True
        
        try:
            # Get or create LLMUsage record
            usage_record = db.query(LLMUsage).filter(
                LLMUsage.document_id == self.document_id
            ).first()
            
            if not usage_record:
                # Create new record
                usage_record = LLMUsage(document_id=self.document_id)
                db.add(usage_record)
            
            # Update specific stats field based on llm_type
            stats = self.get_stats()
            
            if self.llm_type == "text":
                usage_record.text_stats = stats
            elif self.llm_type == "image":
                usage_record.image_stats = stats
            elif self.llm_type == "table":
                usage_record.table_stats = stats
            elif self.llm_type == "chat_agent":
                usage_record.chat_agent_stats = stats
            
            # Recalculate overall stats
            usage_record.overall_stats = self._calculate_overall_stats(usage_record)
            
            db.commit()
            
            print(f"[DB] Saved {self.labels[self.llm_type]} stats for document {self.document_id}")
            
        except Exception as e:
            db.rollback()
            print(f"[ERROR] Failed to save {self.llm_type} stats to DB: {e}")
            raise
        
        finally:
            if close_db:
                db.close()
    
    @staticmethod
    def _calculate_overall_stats(usage_record: LLMUsage) -> Dict[str, Any]:
        """Calculate aggregated overall statistics"""
        total_input = 0
        total_output = 0
        total_cost = 0.0
        
        for stat_field in ['text_stats', 'image_stats', 'table_stats', 'chat_agent_stats']:
            stats = getattr(usage_record, stat_field, {}) or {}
            total_input += stats.get('input_tokens', 0)
            total_output += stats.get('output_tokens', 0)
            total_cost += stats.get('cost_in_inr', 0.0)
        
        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_in_inr": round(total_cost, 2)
        }
    
    def print_summary(self):
        """Print summary of tracked usage"""
        print(f"\n{'='*60}")
        print(f"{self.labels[self.llm_type]} Usage Summary")
        print(f"{'='*60}")
        print(f"LLM Calls: {self.llm_calls}")
        print(f"Input Tokens: {self.input_tokens:,}")
        print(f"Output Tokens: {self.output_tokens:,}")
        print(f"Total Cost: ₹{self.cost_in_inr:.2f}")
        print(f"{'='*60}\n")


class LLMUsageManager:
    """
    Manager for all LLM trackers for a document
    Provides centralized access and batch operations
    """
    
    def __init__(self, document_id: UUID = None):
        """
        Initialize manager with optional document_id
        
        Args:
            document_id: UUID of document being processed
        """
        self.document_id = document_id
        self.trackers: Dict[LLMType, LLMUsageTracker] = {}
    
    def set_document_id(self, document_id: UUID):
        """Set document ID for all trackers"""
        self.document_id = document_id
        for tracker in self.trackers.values():
            tracker.set_document_id(document_id)
    
    def get_tracker(self, llm_type: LLMType, model_name: str = "gemini-2.0-flash-exp") -> LLMUsageTracker:
        """
        Get or create tracker for specific LLM type
        
        Args:
            llm_type: Type of LLM
            model_name: Model being used
            
        Returns:
            LLMUsageTracker instance
        """
        if llm_type not in self.trackers:
            self.trackers[llm_type] = LLMUsageTracker(
                llm_type=llm_type,
                model_name=model_name,
                document_id=self.document_id
            )
        return self.trackers[llm_type]
    
    def save_all_to_db(self, db: Session = None):
        """Save all trackers to database"""
        if not self.document_id:
            print("[WARNING] Cannot save to DB: document_id not set")
            return
        
        close_db = False
        if db is None:
            db = next(get_db())
            close_db = True
        
        try:
            # Get or create record
            usage_record = db.query(LLMUsage).filter(
                LLMUsage.document_id == self.document_id
            ).first()
            
            if not usage_record:
                usage_record = LLMUsage(document_id=self.document_id)
                db.add(usage_record)
            
            # Update all stats
            for llm_type, tracker in self.trackers.items():
                stats = tracker.get_stats()
                if llm_type == "text":
                    usage_record.text_stats = stats
                elif llm_type == "image":
                    usage_record.image_stats = stats
                elif llm_type == "table":
                    usage_record.table_stats = stats
                elif llm_type == "chat_agent":
                    usage_record.chat_agent_stats = stats
            
            # Calculate overall stats
            usage_record.overall_stats = LLMUsageTracker._calculate_overall_stats(usage_record)
            
            db.commit()
            
            print(f"\n[DB] ✓ Saved all LLM usage stats for document {self.document_id}")
            self.print_all_summaries()
            
        except Exception as e:
            db.rollback()
            print(f"[ERROR] Failed to save LLM usage to DB: {e}")
            raise
        
        finally:
            if close_db:
                db.close()
    
    def print_all_summaries(self):
        """Print summary for all trackers"""
        if not self.trackers:
            print("[INFO] No LLM trackers active")
            return
        
        print(f"\n{'='*80}")
        print(f"COMPLETE LLM USAGE REPORT - Document {self.document_id}")
        print(f"{'='*80}")
        
        total_input = 0
        total_output = 0
        total_cost = 0.0
        total_calls = 0
        
        for llm_type, tracker in self.trackers.items():
            stats = tracker.get_stats()
            print(f"\n{stats['label']}:")
            print(f"  Calls: {stats['llm_calls']}")
            print(f"  Input Tokens: {stats['input_tokens']:,}")
            print(f"  Output Tokens: {stats['output_tokens']:,}")
            print(f"  Cost: ₹{stats['cost_in_inr']:.2f}")
            
            total_input += stats['input_tokens']
            total_output += stats['output_tokens']
            total_cost += stats['cost_in_inr']
            total_calls += stats['llm_calls']
        
        print(f"\n{'-'*80}")
        print(f"OVERALL TOTALS:")
        print(f"  Total LLM Calls: {total_calls}")
        print(f"  Total Input Tokens: {total_input:,}")
        print(f"  Total Output Tokens: {total_output:,}")
        print(f"  Total Cost: ₹{total_cost:.2f}")
        print(f"{'='*80}\n")
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get aggregated stats across all trackers"""
        total_input = 0
        total_output = 0
        total_cost = 0.0
        
        for tracker in self.trackers.values():
            stats = tracker.get_stats()
            total_input += stats['input_tokens']
            total_output += stats['output_tokens']
            total_cost += stats['cost_in_inr']
        
        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_in_inr": round(total_cost, 2)
        }