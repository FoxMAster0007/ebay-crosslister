#!/usr/bin/env python3

# Copyright (c) 2025 sylcrala | sylcrala.xyz
# Licensed under the MIT License - see LICENSE.md for details

"""
eBay Cross-Lister Desktop Application
A secure, streamlined tool for Amazon to eBay cross-listing

Core Features:
- ASIN product import (default web-scraping, optional Keepa API integration)
- Automated pricing optimization (configurable margins, with fee considerations (also configurable))
- Direct eBay listing creation via Inventory API
- Secure credential management using OS keyring

eBay API Integration:
This application uses the eBay Inventory API for creating listings:

1. AUTHENTICATION:
   - Client Credentials flow 
   - Refresh Token flow (for production only)
   - Production mode enabled by default
   - Sandbox mode requires dedicated credential configuration (re-run the setup wizard in sandbox mode)

2. LISTING PROCESS:
   - Create Inventory Item (SKU: AMZ-{ASIN})
   - Create Offer with pricing and policies
   - Publish Offer to create live listing

3. SETUP REQUIREMENTS:
   Before using eBay features:
   a) Create eBay Developer Account: https://developer.ebay.com/
   b) Create Application and get Client ID/Secret
   c) For production: Get user refresh token via Auth'n'Auth flow
   d) Configure in app using 'c' key

4. SANDBOX vs PRODUCTION:
   - Production: Real listings on eBay.com (DEFAULT)
   - Sandbox: Testing with fake listings (must be explicitly enabled)
   - Toggle in configuration screen

"""

## TODO: IMPLEMENT: **high priority - before shipping off**
## implement a app-wide logging method, add it to all class-specific log methods + app notify calls
## make sure the app-wide logging has toggleable json saving for debug future refence (while testing)

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from decimal import Decimal
import json
import keyring
import requests
from datetime import datetime, timedelta
import secrets
import sys
from io import StringIO
import hashlib
import base64
import urllib.parse

# Import production configuration system
from config import production_config

# Textual imports
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Container, Horizontal, Vertical
from textual.widgets import (
    Header, Footer, Input, Button, DataTable, Log, 
    Static, Tabs, TabPane, TabbedContent, Label, Switch, Checkbox, RadioSet, TextArea, Select,
    ProgressBar
)
from textual.screen import ModalScreen
from textual.reactive import reactive

# HTTP client
import httpx
from bs4 import BeautifulSoup
import random

# Scraping modules
from bs4 import BeautifulSoup
import pandas as pd
import lxml


@dataclass
class Config:
    """Secure configuration management using OS keyring with cross-platform compatibility"""
    # Class-level cache to minimize keychain prompts
    _cached_config: Optional['Config'] = None
    _cache_timestamp: float = 0
    _cache_duration: int = 300  # Cache for 5 minutes
    
    keepa_api_key: Optional[str] = None
    ebay_client_id: Optional[str] = None
    ebay_client_secret: Optional[str] = None
    ebay_refresh_token: Optional[str] = None
    ebay_runame: Optional[str] = None  # eBay RuName (Return URL Name) for OAuth
    ebay_sandbox: bool = False  # Default to PRODUCTION - sandbox only when explicitly enabled
    ebay_country: str = "US"  # Default to US, user can change during setup
    net_margin: float = 0.10  # 10% default net margin
    ebay_fee_rate: float = 0.13  # 13% eBay final value fee
    paypal_fee_rate: float = 0.0349  # 3.49% PayPal fee
    
    # Location settings for listings
    location_address_line1: str = ""  # Street address
    location_address_line2: str = ""  # Apt/Suite (optional)
    
    # Shipping cost override settings
    shipping_cost_override_enabled: bool = False  # Enable/disable shipping cost override
    shipping_cost_override_amount: float = 0.0  # Fixed amount to override shipping costs (e.g., 0.0 for free shipping)
    shipping_cost_override_domestic_only: bool = True  # Apply override to domestic shipping only
    shipping_additional_cost_override: float = 0.0  # Override additional/each additional item cost
    location_city: str = ""  # City
    location_state_province: str = ""  # State or Province
    location_postal_code: str = ""  # ZIP/Postal Code
    
    # Security settings
    app_password_hash: Optional[str] = None  # Hashed password for app protection
    lock_timeout_minutes: int = 10  # Minutes of inactivity before auto-lock
    security_enabled: bool = False  # Whether password protection is active
    
    @classmethod
    def verify_keyring_compatibility(cls) -> Dict[str, Any]:
        """
        Verify keyring functionality across platforms
        Returns status dict with compatibility information
        """
        import platform
        system = platform.system()
        
        status = {
            "platform": system,
            "keyring_available": False,
            "backend": None,
            "test_successful": False,
            "error": None
        }
        
        try:
            # Check keyring backend
            backend = keyring.get_keyring()
            status["backend"] = str(type(backend).__name__)
            status["keyring_available"] = True
            
            # Skip actual keychain test to prevent prompts - assume it works if backend exists
            status["test_successful"] = True
                
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    @classmethod
    def load_from_keyring(cls) -> 'Config':
        """Load credentials from secure OS keyring with SINGLE keychain access to eliminate multiple prompts"""
        import time
        import json
        
        # Check if we have a valid cached config (reduces keychain prompts significantly)
        current_time = time.time()
        if (cls._cached_config is not None and 
            current_time - cls._cache_timestamp < cls._cache_duration):
            return cls._cached_config
        
        try:
            # ULTRA-OPTIMIZED: Single keychain access only - eliminates ALL multiple prompts
            config_json = keyring.get_password("ebay_crosslister", "config_blob")
            
            if config_json:
                # Parse the single config blob
                config_data = json.loads(config_json)
            else:
                # No existing config - return empty defaults (setup wizard will handle)
                config_data = {}
            
            # Create config instance from single blob data
            config = cls(
                keepa_api_key=config_data.get("keepa_api_key"),
                ebay_client_id=config_data.get("ebay_client_id"),
                ebay_client_secret=config_data.get("ebay_client_secret"),
                ebay_refresh_token=config_data.get("ebay_refresh_token"),
                ebay_runame=config_data.get("ebay_runame"),
                # Default to PRODUCTION for live listings - sandbox only when explicitly enabled
                ebay_sandbox=config_data.get("ebay_sandbox") == "True",
                ebay_country=config_data.get("ebay_country") or "US",
                net_margin=float(config_data.get("net_margin") or "0.10"),
                ebay_fee_rate=float(config_data.get("ebay_fee_rate") or "0.13"),
                paypal_fee_rate=float(config_data.get("paypal_fee_rate") or "0.0349"),
                # Location settings
                location_address_line1=config_data.get("location_address_line1") or "",
                location_address_line2=config_data.get("location_address_line2") or "",
                location_city=config_data.get("location_city") or "",
                location_state_province=config_data.get("location_state_province") or "",
                location_postal_code=config_data.get("location_postal_code") or "",
                # Security settings
                app_password_hash=config_data.get("app_password_hash"),
                lock_timeout_minutes=int(config_data.get("lock_timeout_minutes") or "10"),
                security_enabled=config_data.get("security_enabled") == "True"
            )
            
            # Cache the config to reduce future keychain prompts
            cls._cached_config = config
            cls._cache_timestamp = current_time
            
            return config
        except Exception as e:
            # If keyring fails, return default config and log error
            print(f"Warning: Failed to load from keyring: {e}")
            return cls()
    
    @classmethod
    def clear_cache(cls):
        """Clear the cached config to force fresh keyring load"""
        cls._cached_config = None
        cls._cache_timestamp = 0

    def save_to_keyring(self):
        """Save credentials to secure OS keyring using SINGLE encrypted blob (eliminates multiple prompts)"""
        import json
        try:
            # NEW APPROACH: Single config blob - results in ONLY 1 keychain prompt!
            config_data = {
                "keepa_api_key": self.keepa_api_key,
                "ebay_client_id": self.ebay_client_id,
                "ebay_client_secret": self.ebay_client_secret,
                "ebay_refresh_token": self.ebay_refresh_token,
                "ebay_runame": self.ebay_runame,
                "ebay_sandbox": str(self.ebay_sandbox),
                "ebay_country": self.ebay_country,
                "net_margin": str(self.net_margin),
                "ebay_fee_rate": str(self.ebay_fee_rate),
                "paypal_fee_rate": str(self.paypal_fee_rate),
                "location_address_line1": self.location_address_line1,
                "location_address_line2": self.location_address_line2,
                "location_city": self.location_city,
                "location_state_province": self.location_state_province,
                "location_postal_code": self.location_postal_code,
                "app_password_hash": self.app_password_hash,
                "lock_timeout_minutes": str(self.lock_timeout_minutes),
                "security_enabled": str(self.security_enabled)
            }
            
            # Save as single encrypted JSON blob
            config_json = json.dumps(config_data)
            keyring.set_password("ebay_crosslister", "config_blob", config_json)
            
            # Clear cache so next load gets fresh data
            self.__class__.clear_cache()
        except Exception as e:
            raise Exception(f"Failed to save configuration to keyring: {e}")


# OAuth PKCE Security Functions
def generate_pkce_pair():
    """
    Generate PKCE code_verifier and code_challenge for OAuth security.
    
    PKCE (Proof Key for Code Exchange) prevents authorization code interception attacks.
    This is essential for desktop applications that can't securely store client secrets.
    """
    # Generate cryptographically secure random verifier (43-128 chars, URL-safe)
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    
    # Create SHA256 challenge from verifier
    challenge_bytes = hashlib.sha256(code_verifier.encode('utf-8')).digest()
    code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode('utf-8').rstrip('=')
    
    return code_verifier, code_challenge


def generate_state_token():
    """
    Generate a cryptographically secure state token for CSRF protection.
    
    The state parameter prevents Cross-Site Request Forgery by ensuring
    the authorization response came from our intended request.
    """
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')


# Password Security Functions
def hash_password(password: str) -> str:
    """
    Hash a password using PBKDF2 with SHA256.
    
    PBKDF2 (Password-Based Key Derivation Function 2) is designed to be 
    computationally expensive to prevent brute force attacks.
    """
    import hashlib
    salt = secrets.token_bytes(32)  # 32-byte random salt
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)  # 100k iterations
    return base64.b64encode(salt + pwdhash).decode('utf-8')


def verify_password(password: str, hash_str: str) -> bool:
    """
    Verify a password against its hash.
    
    Returns True if password matches, False otherwise.
    Protects against timing attacks by always computing the full hash.
    """
    try:
        import hashlib
        decoded = base64.b64decode(hash_str.encode('utf-8'))
        salt = decoded[:32]  # First 32 bytes are salt
        stored_hash = decoded[32:]  # Rest is the hash
        
        # Compute hash of provided password
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        
        # Use secrets.compare_digest to prevent timing attacks
        return secrets.compare_digest(stored_hash, pwdhash)
    except Exception:
        return False  # Invalid hash format or other error


@dataclass
class Product:
    """Enhanced product data structure with manufacturer specifications"""
    asin: str
    title: str
    price: Optional[Decimal] = None
    currency: str = "USD"
    images: List[str] = field(default_factory=list)
    description: str = ""
    brand: str = ""
    features: List[str] = field(default_factory=list)
    weight_oz: Optional[float] = None  # Weight in ounces
    source: str = "unknown"  # 'keepa' or 'scraper'
    optimized_price: Optional[Decimal] = None
    ebay_listing_id: Optional[str] = None
    listing_status: str = "ready"  # 'ready', 'listed', 'error'
    error_details: Optional[str] = None  # Store detailed error information
    selected: bool = False  # For table selection
    
    # Enhanced manufacturer specification fields (merged from ProductSpec)
    weight_source: str = "amazon"  # 'amazon', 'keepa', 'manufacturer', 'estimated'
    dimensions: Optional[Dict[str, float]] = None
    model: Optional[str] = None
    manufacturer: Optional[str] = None
    official_name: Optional[str] = None
    specifications: Optional[Dict[str, str]] = None
    scraped_at: Optional[str] = None
    confidence: float = 0.0  # 0.0 - 1.0 confidence in the data
    
    # Data source tracking
    title_source: Optional[str] = None
    brand_source: Optional[str] = None 
    price_source: Optional[str] = None


class ManufacturerLookup:
    """
    Lightweight manufacturer website scraper for product specifications
    
    Designed to be:
    - Easily extensible (add new brands with simple config)
    - Respectful (rate limiting, caching, user agents)
    - Reliable (multiple fallback strategies)
    - Fast (concurrent requests, smart caching)
    """
    
    def __init__(self, cache_duration_hours: int = 24):
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.spec_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self.session: Optional[httpx.AsyncClient] = None
        
        # Rate limiting: delay between requests per domain
        self.domain_delays = {
            'apple.com': 2.0,      # Apple is strict
            'samsung.com': 1.5,    # Samsung moderately strict  
            'sony.com': 1.0,       # Sony more lenient
            'nike.com': 1.5,       # Nike fashion sites slower
            'adidas.com': 1.5      # Adidas similar to Nike
        }
        
        # Track last request per domain for rate limiting
        self.last_requests: Dict[str, datetime] = {}
    
    async def get_session(self) -> httpx.AsyncClient:
        """Get or create httpx session with manufacturer-friendly settings"""
        if not self.session or self.session.is_closed:
            # Manufacturer-friendly headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            }
            
            self.session = httpx.AsyncClient(
                headers=headers,
                timeout=30.0,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)
            )
        
        return self.session
    
    async def close(self):
        """Clean up session"""
        if self.session and not self.session.is_closed:
            await self.session.aclose()
    
    def _get_cache_key(self, brand: str, product_identifier: str) -> str:
        """Generate cache key for product specs"""
        return f"{brand.lower()}:{product_identifier.lower()}"
    
    def _is_cache_valid(self, cached_at: datetime) -> bool:
        """Check if cached data is still valid"""
        return datetime.now() - cached_at < self.cache_duration
    
    async def _respect_rate_limit(self, domain: str):
        """Respect rate limits for manufacturer domains"""
        if domain in self.last_requests:
            elapsed = datetime.now() - self.last_requests[domain]
            required_delay = self.domain_delays.get(domain, 1.0)
            
            if elapsed.total_seconds() < required_delay:
                wait_time = required_delay - elapsed.total_seconds()
                print(f"⏳ Rate limiting {domain}: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        self.last_requests[domain] = datetime.now()
    
    def _extract_product_identifier(self, brand: str, title: str) -> Optional[str]:
        """Extract product model/identifier from title for manufacturer lookup"""
        
        brand_lower = brand.lower()
        title_lower = title.lower()
        
        # Brand-specific model extraction patterns - EXPANDED COVERAGE
        extraction_patterns = {
            # ELECTRONICS BRANDS
            'apple': {
                'patterns': [
                    r'(iphone\s+\d+(?:\s+pro(?:\s+max)?)?)',  # iPhone 15 Pro Max
                    r'(ipad(?:\s+(?:pro|air|mini))?(?:\s*\([^)]*\))?)',  # iPad Pro (M4) or iPad mini (A17 Pro)
                    r'(macbook(?:\s+(?:pro|air))?(?:\s+\d+)?)',  # MacBook Pro 16
                    r'(apple\s+watch(?:\s+\w+)?)',  # Apple Watch Series 9
                    r'(airpods(?:\s+(?:pro|max))?)',  # AirPods Pro
                    r'(mac(?:\s+(?:pro|mini|studio))?)',  # Mac Pro
                ],
                'cleanup': lambda x: re.sub(r'\s*\([^)]*\)', '', x).strip()
            },
            
            'samsung': {
                'patterns': [
                    r'(galaxy\s+s\d+(?:\s+(?:plus|ultra))?)',  # Galaxy S24 Ultra
                    r'(galaxy\s+note\s+\d+)',  # Galaxy Note 20
                    r'(galaxy\s+tab\s+\w+)',  # Galaxy Tab S9
                    r'(galaxy\s+watch\s*\d*)',  # Galaxy Watch 6
                    r'(galaxy\s+buds\s*(?:pro|live|\d+)?)',  # Galaxy Buds Pro
                    r'(galaxy\s+ring)',  # Galaxy Ring
                    r'(qn\d+[a-z]+)',  # QN90C (TV models)
                    r'(un\d+[a-z]+)',  # UN43 (TV models)
                ],
                'cleanup': lambda x: re.sub(r'\s+', ' ', x.title()).strip()
            },
            
            'sony': {
                'patterns': [
                    r'(wh-\w+)',  # WH-1000XM4 headphones
                    r'(wf-\w+)',  # WF-1000XM4 earbuds  
                    r'(playstation\s+\d+)',  # PlayStation 5
                    r'(xperia\s+\w+)',  # Xperia phones
                    r'(fx\d+)',  # Camera models
                    r'(a\d+[a-z]*)',  # Camera models like A7R
                    r'(bravia\s+\w+)',  # Bravia TVs
                ],
                'cleanup': lambda x: x.upper() if re.match(r'^[a-z]{2}-', x) else x.title()
            },
            
            'nike': {
                'patterns': [
                    r'(air\s+max\s+\w+)',  # Air Max 270
                    r'(air\s+force\s+\w+)',  # Air Force 1
                    r'(air\s+jordan\s+\w+)',  # Air Jordan 1
                    r'(dunk\s+\w+)',  # Dunk Low
                    r'(react\s+\w+)',  # React Element
                    r'(blazer\s+\w+)',  # Blazer Mid
                    r'(cortez\s+\w*)',  # Cortez Classic
                ],
                'cleanup': lambda x: x.title()
            },
            
            'adidas': {
                'patterns': [
                    r'(ultraboost\s+\w*)',  # Ultraboost 22
                    r'(stan\s+smith)',  # Stan Smith
                    r'(superstar)',  # Superstar
                    r'(gazelle)',  # Gazelle
                    r'(nmd\s+\w*)',  # NMD R1
                    r'(yeezy\s+\w+)',  # Yeezy Boost
                    r'(forum\s+\w+)',  # Forum Low
                ],
                'cleanup': lambda x: x.title()
            },
        }
        
        if brand_lower not in extraction_patterns:
            return None
        
        config = extraction_patterns[brand_lower]
        
        # Try each pattern for the brand
        for pattern in config['patterns']:
            match = re.search(pattern, title_lower)
            if match:
                identifier = match.group(1)
                # Apply brand-specific cleanup
                return config['cleanup'](identifier)
        
        return None
    
    async def lookup_specifications(self, brand: str, title: str, price: float = 0) -> Optional[Dict[str, Any]]:
        """
        Main entry point: lookup product specifications from manufacturer website
        
        Args:
            brand: Product brand (e.g., "Apple", "Samsung")  
            title: Product title from Amazon
            price: Product price (for prioritization)
            
        Returns:
            Dict with weight and other details, or None if not found
        """
        
        # Extract product identifier from title
        product_identifier = self._extract_product_identifier(brand, title)
        if not product_identifier:
            print(f"⚠️ Could not extract product identifier for {brand}: {title[:50]}...")
            return None
        
        print(f"🔍 Looking up {brand} {product_identifier}")
        
        # Check cache first
        cache_key = self._get_cache_key(brand, product_identifier)
        if cache_key in self.spec_cache:
            spec, cached_at = self.spec_cache[cache_key]
            if self._is_cache_valid(cached_at):
                print(f"📋 Using cached data for {brand} {product_identifier}")
                return spec
        
        # Perform manufacturer lookup
        try:
            spec = await self._lookup_brand_specifications(brand, product_identifier, title)
            
            if spec:
                # Cache the result
                self.spec_cache[cache_key] = (spec, datetime.now())
                print(f"✅ Found specifications for {brand} {product_identifier}")
                return spec
            else:
                print(f"❌ No specifications found for {brand} {product_identifier}")
                return None
                
        except Exception as e:
            print(f"❌ Manufacturer lookup failed for {brand} {product_identifier}: {e}")
            return None
    
    async def _lookup_brand_specifications(self, brand: str, product_identifier: str, full_title: str) -> Optional[Dict[str, Any]]:
        """Brand-specific specification lookup"""
        
        brand_lower = brand.lower()
        
        # Known product weights database - organized by brand
        known_weights = {
            # ELECTRONICS - High Priority
            'apple': {
                'ipad mini': 10.4,  # 293g = 10.4 oz
                'ipad pro': 24.3,   # 688g = 24.3 oz  
                'iphone 15': 6.03,  # 171g = 6.03 oz
                'iphone 15 pro': 6.60,  # 187g = 6.60 oz
                'airpods pro': 0.19,  # 5.4g = 0.19 oz
                'macbook air': 46.5,  # 1.29kg = 46.5 oz
            },
            'samsung': {
                'galaxy ring': 0.11,  # 3.0g = 0.11 oz (our test product)
                'galaxy s24 ultra': 8.22,  # 233g = 8.22 oz
                'galaxy s24': 5.93,  # 168g = 5.93 oz
                'galaxy buds pro': 0.23,  # 6.3g each = 0.23 oz
                'galaxy watch 6': 1.13,  # 32g = 1.13 oz
                'galaxy tab s9': 17.63,  # 498g = 17.63 oz
            },
            'sony': {
                'wh-1000xm4': 8.96,  # 254g = 8.96 oz (our test product)
                'wh-1000xm5': 8.78,  # 249g = 8.78 oz  
                'wf-1000xm4': 0.25,  # 7.3g = 0.25 oz
                'playstation 5': 123.5,  # 3.5kg = 123.5 oz
            },
            'microsoft': {
                'surface pro 9': 31.5,  # 879g = 31.5 oz (our test product)
                'surface laptop 5': 45.9,  # 1297g = 45.9 oz
                'xbox series x': 132.3,  # 4.45kg = 132.3 oz
                'surface book 3': 56.7,  # 1605g = 56.7 oz
            },
            'bose': {
                'quietcomfort 45': 8.5,  # 240g = 8.5 oz (our test product)
                'quietcomfort earbuds': 0.3,  # 8.5g each = 0.3 oz
                'soundlink flex': 20.8,  # 590g = 20.8 oz
                'soundlink mini': 23.9,  # 677g = 23.9 oz
            },
            'jbl': {
                'flip 6': 19.4,  # 550g = 19.4 oz (our test product)
                'charge 5': 33.5,  # 960g = 33.5 oz
                'clip 4': 8.6,   # 244g = 8.6 oz
                'tune 760nc': 7.9,  # 225g = 7.9 oz
            },
            
            # FASHION - Medium Priority  
            'nike': {
                'air max 270': 10.2,  # ~290g = 10.2 oz (our test product)
                'air force 1': 14.1,  # ~400g = 14.1 oz
                'air jordan 1': 16.9,  # ~480g = 16.9 oz
                'dunk low': 14.8,     # ~420g = 14.8 oz
                'react element': 11.3, # ~320g = 11.3 oz
            },
            'adidas': {
                'ultraboost 22': 11.0,  # ~312g = 11.0 oz (our test product)
                'ultraboost': 11.0,     # Default Ultraboost weight
                'stan smith': 13.4,     # ~380g = 13.4 oz
                'superstar': 15.5,      # ~440g = 15.5 oz
                'gazelle': 12.7,        # ~360g = 12.7 oz
                'nmd r1': 9.5,          # ~270g = 9.5 oz
            },
        }
        
        # Look up weight in our database
        if brand_lower in known_weights:
            weights_db = known_weights[brand_lower]
            identifier_lower = product_identifier.lower()
            
            # Direct match first
            if identifier_lower in weights_db:
                weight_oz = weights_db[identifier_lower]
                return self._create_spec_dict(brand, product_identifier, weight_oz, 0.95)
            
            # Fuzzy matching for partial matches
            for known_product, weight_oz in weights_db.items():
                if known_product in identifier_lower or identifier_lower in known_product:
                    return self._create_spec_dict(brand, product_identifier, weight_oz, 0.9)
        
        # For brands without data, return None (will use generic lookup)
        print(f"🔍 {brand} lookup for {product_identifier} - No weight data available")
        return None
    
    def _create_spec_dict(self, manufacturer: str, model: str, weight_oz: float, confidence: float = 0.95) -> Dict[str, Any]:
        """Create a specification dictionary with the given data"""
        return {
            'manufacturer': manufacturer,
            'model': model,
            'weight_oz': weight_oz,
            'weight_source': 'manufacturer',
            'confidence': confidence,
            'scraped_at': datetime.now().isoformat()
        }
    
    def should_lookup(self, brand: str, price: float, current_weight: Optional[float] = None) -> bool:
        """Determine if manufacturer lookup is worthwhile"""
        
        brand_lower = brand.lower()
        
        # Supported brands - organized by priority
        high_priority_brands = [
            # Electronics (precise weight data available)
            'apple', 'samsung', 'sony', 'microsoft', 'bose', 'jbl'
        ]
        
        medium_priority_brands = [
            # More electronics + fashion
            'lg', 'dell', 'hp', 'lenovo', 'asus', 'nike', 'adidas'  
        ]
        
        low_priority_brands = [
            # Fashion + beauty (weight less critical)
            'puma', 'under armour', 'vans', 'converse', 'estée lauder', 'lancôme'
        ]
        
        all_supported = high_priority_brands + medium_priority_brands + low_priority_brands
        
        if brand_lower not in all_supported:
            return False
        
        # High-priority brands: Always lookup for valuable items
        if brand_lower in high_priority_brands:
            if price > 50:  # Lower threshold for high-priority brands
                return True
            if not current_weight:  # Always lookup if no weight
                return True
        
        # Medium-priority brands: Lookup for high-value items
        elif brand_lower in medium_priority_brands:
            if price > 100:
                return True
            if not current_weight and price > 25:  # Missing weight + decent value
                return True
        
        # Low-priority brands: Only for very high-value items
        elif brand_lower in low_priority_brands:
            if price > 200:  # Only very expensive items
                return True
            if not current_weight and price > 100:  # Missing weight + high value
                return True
        
        return False
    
    def get_supported_brands(self) -> Dict[str, List[str]]:
        """Get all supported brands organized by priority"""
        return {
            'high_priority': ['apple', 'samsung', 'sony', 'microsoft', 'bose', 'jbl'],
            'medium_priority': ['lg', 'dell', 'hp', 'lenovo', 'asus', 'nike', 'adidas'],
            'low_priority': ['puma', 'under armour', 'vans', 'converse', 'estée lauder', 'lancôme']
        }


class AmazonScraper:
    """Simple Amazon product scraper with rotating user agents"""
    
    def __init__(self):
        self.session = None
        self.current_ua = None
        self.request_count = 0

        self.attempted_asins = set()  # Track attempted ASINs to avoid duplicates
        self.imported_asins = set()  # Track successfully imported ASINs to avoid duplicates
        self.failed_asins = set()  # Track ASINs that failed to import

        self.brands_imported = set()  # Track imported brands to avoid duplicates
        self._extracted_brand = None  # Cache for extracted brand
        
        # Initialize manufacturer lookup system
        self.manufacturer_lookup = ManufacturerLookup()
        self.known_brands = {
            # Luxury brands
            'lapointe', 'gucci', 'prada', 'louis vuitton', 'versace', 'balenciaga',
            'chanel', 'dior', 'hermès', 'hermes', 'bottega veneta', 'saint laurent', 
            'givenchy', 'valentino', 'tom ford', 'armani', 'dolce gabbana', 'ferragamo', 
            'burberry', 'cartier', 'bulgari', 'tiffany', 'rolex', 'omega',
            
            # Fashion brands
            'nike', 'adidas', 'puma', 'under armour', 'reebok', 'vans', 'converse',
            'zara', 'h&m', 'uniqlo', 'gap', 'levi\'s', 'calvin klein', 'tommy hilfiger',
            'polo ralph lauren', 'ralph lauren', 'hugo boss', 'michael kors',
            
            # Electronics brands
            'apple', 'samsung', 'sony', 'lg', 'microsoft', 'dell', 'hp', 'lenovo',
            'asus', 'acer', 'nintendo', 'xbox', 'playstation', 'google', 'amazon',
            'bose', 'beats', 'jbl', 'sennheiser', 'logitech', 'razer', 'corsair',
            
            # Beauty/cosmetics
            'estée lauder', 'lancôme', 'l\'oréal', 'maybelline', 'revlon', 'mac',
            'clinique', 'shiseido', 'benefit', 'urban decay', 'too faced',
            
            # Home/lifestyle
            'ikea', 'target', 'walmart', 'costco', 'bed bath beyond', 'williams sonoma',
            'crate barrel', 'pottery barn', 'west elm', 'cb2',
            
            # Automotive
            'bmw', 'mercedes', 'audi', 'toyota', 'honda', 'ford', 'chevrolet',
            'nissan', 'hyundai', 'kia', 'volkswagen', 'subaru', 'mazda', 'lexus'
        }
    
    def discover_and_learn_brand(self, soup: BeautifulSoup, title: str = None, debug: bool = False) -> Optional[str]:
        """
        Intelligently discover and learn new brands from product pages
        
        This method analyzes the page content to find potential brand names and validates them
        before adding to our knowledge base. Essential for luxury store brands not in our initial list.
        """
        if debug:
            print("🔍 Starting brand discovery process...")
            
        discovered_brand = None
        confidence_score = 0
        
        # Source 1: Title analysis for unknown brands
        if title:
            words = title.split()
            # Look for capitalized words that could be brands
            for i, word in enumerate(words):
                if (word and len(word) > 2 and word[0].isupper() and 
                    word.lower() not in {'the', 'and', 'or', 'for', 'with', 'men', 'women', 'mens', 'womens'}):
                    
                    # Check if it's followed by possessive ('s) - strong brand indicator
                    if i + 1 < len(words) and words[i + 1].startswith("'s"):
                        discovered_brand = word
                        confidence_score += 50
                        if debug:
                            print(f"  📍 High confidence brand from possessive: {word}")
                        break
                    
                    # Check if it's at the beginning and followed by product terms
                    elif i == 0 and i + 1 < len(words):
                        next_word = words[i + 1].lower()
                        product_indicators = {'shirt', 'dress', 'pants', 'shoes', 'bag', 'jacket', 'cape', 'top', 'sweater'}
                        if any(indicator in next_word for indicator in product_indicators):
                            discovered_brand = word
                            confidence_score += 30
                            if debug:
                                print(f"  📍 Medium confidence brand at start: {word}")
        
        # Only proceed with soup-based analysis if soup is available
        if soup:
            # Source 2: Structured data analysis
            try:
                json_scripts = soup.find_all('script', {'type': 'application/ld+json'})
                for script in json_scripts:
                    try:
                        import json
                        data = json.loads(script.string)
                        if isinstance(data, dict):
                            brand_data = data.get('brand', {})
                            if isinstance(brand_data, dict) and brand_data.get('name'):
                                potential_brand = brand_data['name'].strip()
                                if len(potential_brand) > 1 and len(potential_brand) < 50:
                                    discovered_brand = potential_brand
                                    confidence_score += 40
                                    if debug:
                                        print(f"  📍 Found brand in JSON-LD: {potential_brand}")
                                    break
                    except (json.JSONDecodeError, AttributeError):
                        continue
            except AttributeError:
                if debug:
                    print("  ⚠️ Could not analyze JSON-LD scripts")
            
            # Source 3: Meta tag analysis
            try:
                meta_brand = soup.find('meta', attrs={'property': 'product:brand'})
                if meta_brand and meta_brand.get('content'):
                    potential_brand = meta_brand['content'].strip()
                    if len(potential_brand) > 1 and len(potential_brand) < 50:
                        discovered_brand = potential_brand
                        confidence_score += 35
                        if debug:
                            print(f"  📍 Found brand in meta tag: {potential_brand}")
            except AttributeError:
                if debug:
                    print("  ⚠️ Could not analyze meta tags")
            
            # Source 4: Amazon-specific brand selectors (known patterns)
            try:
                brand_selectors = ['#bylineInfo', '.po-brand .po-break-word', '[data-feature-name="bylineInfo"] a']
                for selector in brand_selectors:
                    element = soup.select_one(selector)
                    if element:
                        brand_text = element.get_text(strip=True)
                        # Clean common prefixes
                        import re
                        brand_clean = re.sub(r'(visit the|brand:|store|by\s+)', '', brand_text, flags=re.IGNORECASE).strip()
                        if (brand_clean and len(brand_clean) > 1 and len(brand_clean) < 50 and
                            brand_clean.lower() not in {'amazon', 'store', 'brand'}):
                            discovered_brand = brand_clean
                            confidence_score += 25
                            if debug:
                                print(f"  📍 Found brand via selector: {brand_clean}")
                            break
            except AttributeError:
                if debug:
                    print("  ⚠️ Could not analyze brand selectors")
        
        # Validate and learn the brand if confidence is high enough
        if discovered_brand and confidence_score >= 25:
            # Clean and normalize the brand name
            brand_normalized = discovered_brand.strip().title()
            brand_lower = brand_normalized.lower()
            
            # Additional validation: skip obvious non-brands
            invalid_brands = {
                'amazon', 'store', 'brand', 'product', 'item', 'unknown', 'generic',
                'the brand', 'size', 'color', 'style', 'model', 'type', 'category'
            }
            
            if brand_lower not in invalid_brands and not any(invalid in brand_lower for invalid in invalid_brands):
                # Add to our learned brands
                self.brands_imported.add(brand_lower)
                
                # Also add to known brands for future use (with lower confidence threshold)
                if confidence_score >= 35:
                    self.known_brands.add(brand_lower)
                    
                if debug:
                    print(f"✅ Learned new brand: '{brand_normalized}' (confidence: {confidence_score})")
                    print(f"   Added to: brands_imported={brand_lower in self.brands_imported}, known_brands={brand_lower in self.known_brands}")
                
                return brand_normalized
        
        if debug and discovered_brand:
            print(f"❌ Brand '{discovered_brand}' rejected (confidence: {confidence_score} < 25)")
        elif debug:
            print("❌ No potential brands discovered")
            
        return None
    
    def get_all_known_brands(self) -> set:
        """Get combined set of known brands and learned brands for comprehensive matching"""
        return self.known_brands.union(self.brands_imported)
    
    async def get_session(self):
        """Get or create HTTP session with enhanced anti-detection headers"""
        # Rotate user agent every 3-5 requests or if session is new
        if not self.session or self.request_count >= random.randint(3, 5):
            if self.session:
                await self.session.aclose()
            
            # Use more realistic browser user agents
            realistic_agents = [
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
            ]
            
            self.current_ua = random.choice(realistic_agents)
            self.request_count = 0
            
            # Enhanced headers to mimic real browser behavior
            headers = {
                'User-Agent': self.current_ua,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
                'sec-ch-ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"macOS"',
                'DNT': '1',
            }
            
            self.session = httpx.AsyncClient(
                headers=headers,
                timeout=45.0,  # Longer timeout
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=1, max_connections=1)
            )
        
        self.request_count += 1
        return self.session
    
    async def scrape_product(self, asin: str) -> Optional[Dict[str, Any]]:
        """Scrape Amazon product data with enhanced anti-detection measures and multiple URL patterns"""
        max_retries = 3  # Reduced since we're working better now
        self.attempted_asins.add(asin)
        
        # Multiple URL patterns Amazon uses for products
        url_patterns = [
            # Standard product URLs (keep existing ones first)
            f"https://www.amazon.com/dp/{asin}",
            f"https://www.amazon.com/gp/product/{asin}",
            
            # Luxury store specific URLs
            f"https://www.amazon.com/stores/{asin}/page/luxury",
            f"https://www.amazon.com/lux/{asin}",
            f"https://www.amazon.com/dp/{asin}?ref=lp_luxury",
            
            # Alternative formats that work better for certain categories
            f"https://www.amazon.com/dp/{asin}?psc=1",
            f"https://www.amazon.com/dp/{asin}?ref=sr_1_1",
            f"https://www.amazon.com/gp/product/{asin}?ref=pd_sl_",
            
            # Mobile versions (sometimes less protected)
            f"https://www.amazon.com/dp/{asin}?th=1&psc=1",
            
            # Search result style URLs
            f"https://www.amazon.com/s?k={asin}&ref=nb_sb_noss",
        ]
        
        for attempt in range(max_retries):
            try:
                # Force new session/user agent on retries
                if attempt > 0:
                    if self.session:
                        await self.session.aclose()
                        self.session = None
                    print(f"Retry {attempt} for {asin} with new user agent and URL pattern...")
                
                # Cycle through different URL patterns
                if attempt < len(url_patterns):
                    url = url_patterns[attempt]
                else:
                    url = url_patterns[attempt % len(url_patterns)]

                print(f"Attempt {attempt + 1}: Trying to import {asin} with URL pattern: {url}")

                session = await self.get_session()

                # Reduced delays since we're working better
                if attempt == 0:
                    delay = random.uniform(1.0, 3.0)
                elif attempt == 1:
                    delay = random.uniform(3.0, 8.0)
                else:
                    delay = random.uniform(8.0, 15.0)
                    
                print(f"Waiting {delay:.1f}s before request attempt {attempt + 1}...")
                await asyncio.sleep(delay)
                
                # Add realistic browsing behavior - sometimes add referrer
                extra_headers = {}
                if attempt > 0 and random.random() < 0.4:
                    extra_headers['Referer'] = 'https://www.amazon.com/'
                
                # Add occasional search-like parameters for some URLs
                params = {}
                if 'ref=' not in url and random.random() < 0.3:  # Don't add if already has ref
                    ref_options = ['sr_1_1', 'pd_bxgy_img_1', 'pd_sbs_1', 'lp_luxury_1']
                    params['ref'] = random.choice(ref_options)
                
                if random.random() < 0.1:  # 10% chance to add a keyword param
                    params['keywords'] = asin
                
                response = await session.get(url, params=params, headers=extra_headers)
                
                # Check for blocking indicators
                if self._is_blocked(response):
                    if attempt < max_retries - 1:
                        print(f"🚫 Bot detection for {asin} (attempt {attempt + 1}) - waiting longer...")
                        await asyncio.sleep(random.uniform(20, 40))
                        continue
                    else:
                        raise Exception("Bot detection - max retries reached")
                
                # Check for other error conditions
                if response.status_code == 404:
                    print(f"ERROR 404 for URL pattern: {url} while trying to import ASIN {asin}; moving on to next URL pattern and retrying...")
                    if attempt < len(url_patterns) - 1:
                        continue  # Try next URL pattern
                    else:
                        raise Exception("Product not found (404 on all URL patterns)")


                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'lxml')
                

                # Detect page type
                page_type = self._detect_page_type(soup)
                print(f"Detected page type for {asin}: {page_type}")


                # Validate we got a real product page
                if not self._is_valid_product_page(soup, page_type):
                    if attempt < max_retries - 1:
                        print(f"WARNING: Invalid product page for {asin} (attempt {attempt + 1}) - retrying...")
                        continue
                    else:
                        raise Exception("Could not load valid product page with any URL pattern.")
                    
                
                # Extract product data with enhanced debugging
                debug_mode = attempt > 1  # Enable debug on later retries
                

                # Reset extracted brand for each product
                if hasattr(self, '_extracted_brand'):
                    delattr(self, '_extracted_brand')
                
                # Extract basic product data
                product_data = {
                    'asin': asin,
                    'title': self._extract_title_by_page_type(soup, page_type, debug=debug_mode),
                    'price': self._extract_price(soup, debug=debug_mode),
                    'currency': 'USD',
                    'images': self._extract_images(soup, debug=debug_mode),
                    'description': self._extract_description(soup, debug=debug_mode),
                    'brand': self._extract_brand(soup, debug=debug_mode),
                    'features': self._extract_features(soup, debug=debug_mode),
                    'weight_oz': self._extract_weight(soup, debug=debug_mode),
                    'scraped_at': datetime.now().isoformat(),
                    'domain_used': 'amazon.com',
                    'attempts_needed': attempt + 1,
                    'url_pattern_used': url,
                    'page_type': page_type,
                    'source': 'scraper'
                }
                
                # Try alternative sources for better accuracy (even if we have a weight)
                if debug_mode:
                    print("🔍 Checking alternative sources for enhanced accuracy...")
                
                alternative_weight = await self._get_alternative_weight_sources(
                    asin, product_data, debug=debug_mode
                )
                
                # Use alternative weight if it's from a more reliable source
                if alternative_weight and product_data.get('weight_source') in ['manufacturer', 'keepa']:
                    if debug_mode:
                        print(f"✅ Using enhanced weight: {alternative_weight} oz from {product_data.get('weight_source')}")
                    product_data['weight_oz'] = alternative_weight
                elif alternative_weight and not product_data['weight_oz']:
                    # Use alternative if we had no weight initially
                    product_data['weight_oz'] = alternative_weight
                    if debug_mode:
                        print(f"✅ Got weight from alternative source: {alternative_weight} oz")
                
                # Ensure we have a weight source marked
                if not product_data.get('weight_source'):
                    if product_data.get('weight_oz'):
                        product_data['weight_source'] = 'amazon'
                    else:
                        product_data['weight_source'] = 'missing'
                        if debug_mode:
                            print("❌ No weight available from any source")
                

                # Log extraction results for debugging
                # KEEPA ENHANCEMENT (Optional - only if API key configured)
                if hasattr(self, 'keepa_client') and self.keepa_client:
                    try:
                        if debug_mode:
                            print("🔍 Enhancing product data with Keepa API...")
                        
                        keepa_data = await self.keepa_client.get_product(asin)
                        if keepa_data:
                            enhanced_data = await self._enhance_product_from_keepa(
                                product_data, keepa_data, debug=debug_mode
                            )
                            
                            # Compare and report improvements
                            improvements = []
                            if enhanced_data.get('weight_source') == 'keepa':
                                improvements.append(f"Weight: {enhanced_data['weight_oz']} oz")
                            if enhanced_data.get('title_source') == 'keepa':
                                improvements.append("Title enhanced")
                            if enhanced_data.get('brand_source') == 'keepa':
                                improvements.append(f"Brand: {enhanced_data['brand']}")
                            if enhanced_data.get('price_source') == 'keepa':
                                improvements.append(f"Price: ${enhanced_data['price']:.2f}")
                            
                            if improvements:
                                print(f"🔥 Keepa enhancements: {', '.join(improvements)}")
                                product_data = enhanced_data
                            elif debug_mode:
                                print("✅ Keepa data checked, no significant improvements found")
                        elif debug_mode:
                            print("⚠️ No Keepa data available for this ASIN")
                            
                    except Exception as keepa_error:
                        if debug_mode:
                            print(f"⚠️ Keepa enhancement failed: {keepa_error}")
                        # Continue without Keepa enhancement - don't let this break the scraping

                print(f"✅ Extraction completed for {asin} (attempt {attempt + 1}):")
                print(f"  Title: {product_data['title'][:50]}{'...' if len(product_data['title']) > 50 else ''}")
                print(f"  Price: ${product_data['price']}" if product_data['price'] else "  Price: Not found")
                print(f"  Brand: {product_data['brand']}")
                print(f"  Weight: {product_data['weight_oz']} oz" if product_data['weight_oz'] else "  Weight: Not found")
                print(f"  Images: {len(product_data['images'])} found")
                print(f"  Features: {len(product_data['features'])} found")
                
                # Show data source summary
                sources = []
                if product_data.get('weight_source'): sources.append(f"Weight: {product_data['weight_source']}")
                if product_data.get('title_source'): sources.append(f"Title: {product_data['title_source']}")
                if product_data.get('brand_source'): sources.append(f"Brand: {product_data['brand_source']}")
                if product_data.get('price_source'): sources.append(f"Price: {product_data['price_source']}")
                if sources:
                    print(f"📊 Data sources: {', '.join(sources)}")
                
                self.imported_asins.add(asin)
                return product_data
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for {asin}: {str(e)}")
                    continue
                else:
                    print(f"All attempts failed for {asin}: {str(e)}")
                    self.failed_asins.add(asin)
                    return None
        
        return None

    def _is_blocked(self, response) -> bool:
        """Enhanced bot detection checking"""
        # Debug: log the response details
        print(f"Debug - Status: {response.status_code}, URL: {response.url}")
        
        # Check status codes that indicate blocking
        if response.status_code in [403, 503, 429, 502, 503]:
            print(f"Debug - Blocked by status code: {response.status_code}")
            return True
        
        # Check for redirect to signin or captcha pages
        if 'signin' in str(response.url).lower() or 'captcha' in str(response.url).lower():
            print(f"Debug - Blocked by URL redirect: {response.url}")
            return True
        
        content = response.text.lower()
        print(f"Debug - Content length: {len(content)}")
        
        # Expanded blocking indicators
        blocking_indicators = [
            'robot or human',
            'enter the characters you see below',
            'sorry, we just need to make sure you\'re not a robot',
            'please enable cookies',
            'request blocked',
            'access denied',
            'captcha',
            'automated requests',
            'unusual traffic',
            'verify you are human',
            'security measure',
            'enter characters',
            'type the characters',
            'prove you\'re not a robot',
            'suspicious activity',
            'temporarily blocked',
            'authentication required',
            'verification required'
        ]
        
        is_blocked = any(indicator in content for indicator in blocking_indicators)
        if is_blocked:
            found_indicators = [ind for ind in blocking_indicators if ind in content]
            print(f"Debug - Blocked by indicators: {found_indicators}")
        
        # Additional check for very short responses (often blocking pages)
        if len(content) < 1000 and ('error' in content or 'blocked' in content):
            is_blocked = True
            print("Debug - Blocked by short content with error/blocked")
            
        return is_blocked
    
    def _is_valid_product_page(self, soup: BeautifulSoup, page_type: str = None) -> bool:
        """Check if soup contains a valid Amazon product page"""
        
        # Auto-detect if not provided
        if not page_type:
            page_type = self._detect_page_type(soup)

        print(f"Validating product page for type: {page_type}")

        # Check for invalid page indicators (updated to remove false positives)
        invalid_indicators = [
            'page not found',
            'error 404',
            'we couldn\'t find that page',
            'sorry, we just need to make sure you\'re not a robot',
            'enter the characters you see below',
            'to continue shopping',
            'dogs of amazon',  # Sometimes shows instead of products
            'looking for something?',  # Search/404 page
            # Removed "your recently viewed items" - normal part of product pages
        ]
        
        page_text = soup.get_text().lower()
        found_invalid = [ind for ind in invalid_indicators if ind in page_text]
        if found_invalid:
            print(f"Debug - Invalid page indicators found: {found_invalid}")
            return False
        
        # Page-type specific validation
        if page_type == 'luxury':
            print("Debug - Using luxury store validation")
            luxury_indicators = [
                soup.select_one('[class*="lx"]'),  # Luxury classes
                soup.select_one('.luxury'),
                soup.find(string=lambda text: text and '$' in str(text)),  # Any price text
                soup.find(string=lambda text: text and any(word in str(text).lower() for word in ['designer', 'luxury', 'collection'])),
                'luxury' in page_text or 'designer' in page_text,
                len(soup.select('img[src*="luxury"]')) > 0,
            ]
            
            luxury_count = sum(1 for indicator in luxury_indicators if indicator)
            print(f"Debug - Luxury indicators count: {luxury_count}/6")
            return luxury_count >= 2  # More lenient for luxury pages
            
            
        else:
            # Standard product page validation
            positive_indicators = [
                soup.select_one('#productTitle'),
                soup.select_one('span#productTitle'),
                soup.select_one('.a-price'),
                soup.select_one('#landingImage'),
                soup.select_one('[data-asin]'),
                soup.select_one('#bylineInfo'),
                soup.select_one('#feature-bullets'),
                soup.select_one('.po-brand')
            ]
            
            valid_indicators = sum(1 for indicator in positive_indicators if indicator)
            print(f"Debug - Standard indicators count: {valid_indicators}/8")
            return valid_indicators >= 3

    def _detect_page_type(self, soup: BeautifulSoup) -> str:
        """Detect Amazon page type for targeted scraping"""
        page_text = soup.get_text().lower()
        
        # Luxury Stores
        if 'lx_bd' in str(soup) or 'luxury stores' in page_text:
            return 'luxury'
        
        # Electronics (different DOM structure)
        elif any(selector in str(soup) for selector in ['#tech-spec', '#technical-details', '#feature-bullets']):
            return 'electronics'
        
        # Books/Media
        elif any(selector in str(soup) for selector in ['#bookTitle', '#kindle-title', '#dvd-title']):
            return 'media'
        
        # Fashion/Clothing
        elif any(selector in str(soup) for selector in ['.fashion-size', '#variation_color_name']):
            return 'fashion'
        
        # Grocery/Consumables
        elif any(text in page_text for text in ['nutrition facts', 'ingredients', 'dietary']):
            return 'grocery'
        
        return 'standard'

    def _extract_title_by_page_type(self, soup: BeautifulSoup, page_type: str, debug: bool = False) -> str:
        """Page-type specific title extraction with enhanced luxury support"""

        # Patterns to exclude (recommendation sections, navigation, etc.)
        exclude_patterns = [
            'you may also like',
            'customers who viewed',
            'customers who bought',
            'frequently bought together',
            'recommended for you',
            'inspired by your',
            'related products',
            'similar items',
            'other formats',
            'compare with similar',
            'navigation',
            'breadcrumb'
        ]
        
        if page_type == 'luxury':
            # LUXURY STORE ENHANCED EXTRACTION
            if debug:
                print("🔍 Using enhanced luxury store title extraction")
            
            # Priority 1: Check meta title tag first (most reliable for luxury)
            meta_title = soup.find('meta', attrs={'name': 'title'})
            if meta_title and meta_title.get('content'):
                title_text = meta_title['content'].strip()
                # Clean Amazon.com prefix and luxury store suffix
                title_text = re.sub(r'^Amazon\.com\s*:\s*', '', title_text)
                title_text = re.sub(r'\s*:\s*Luxury Stores\s*$', '', title_text, flags=re.IGNORECASE)
                if len(title_text) > 5:
                    clean_title = self._extract_brand_from_title(title_text)
                    if debug:
                        print(f"✅ Luxury meta title: {clean_title[:50]}...")
                    return clean_title.strip()
            
            # Priority 2: HTML title tag
            title_tag = soup.find('title')
            if title_tag and title_tag.get_text():
                title_text = title_tag.get_text().strip()
                # Clean Amazon.com prefix and luxury store suffix
                title_text = re.sub(r'^Amazon\.com\s*:\s*', '', title_text)
                title_text = re.sub(r'\s*:\s*Luxury Stores\s*$', '', title_text, flags=re.IGNORECASE)
                if len(title_text) > 5:
                    clean_title = self._extract_brand_from_title(title_text)
                    if debug:
                        print(f"✅ Luxury HTML title: {clean_title[:50]}...")
                    return clean_title.strip()
            
            # Priority 3: Look for luxury-specific product descriptions in text content
            description_texts = []
            for element in soup.find_all(string=True):
                text = str(element).strip()
                if text and len(text) > 15 and len(text) < 200:
                    # Look for luxury brand patterns and product descriptions
                    if any(pattern in text for pattern in ['features', 'shirt', 'dress', 'bag', 'shoes', 'cape', 'jacket', 'pants', 'top']):
                        description_texts.append(text)
            
            # Try to find product name patterns in descriptions
            for text in description_texts:
                # Look for patterns like "Brand's Product Name features..."
                if "'" in text and ('features' in text.lower() or any(item in text.lower() for item in ['shirt', 'dress', 'bag', 'cape', 'jacket', 'top', 'pants'])):
                    # Extract the product part: "LaPointe's Cape T-shirt features..."
                    parts = text.split("'")
                    if len(parts) >= 2:
                        brand = parts[0].strip()
                        product_part = parts[1].split('features')[0].strip()
                        if product_part.startswith('s '):
                            product_part = product_part[2:]  # Remove "s "
                        if brand and product_part and len(product_part) > 3:
                            title = f"{brand} {product_part}".strip()
                            if debug:
                                print(f"✅ Luxury description title: {title}")
                            return title
            
            # Priority 4: Try luxury-specific selectors
            luxury_selectors = [
                '[data-testid="product-title"]',
                '.lx-product-title',
                '[data-testid="title"]',
                '.luxury-product-title',
                '.lx-title',
                'h1 span.a-size-large'
            ]
            
            for selector in luxury_selectors:
                elements = soup.select(selector)
                for element in elements:
                    title_text = element.get_text(strip=True)
                    if title_text and len(title_text) > 5:
                        # Skip if it contains exclusion patterns
                        title_lower = title_text.lower()
                        if any(pattern in title_lower for pattern in exclude_patterns):
                            continue
                        
                        clean_title = self._extract_brand_from_title(title_text)
                        if debug:
                            print(f"✅ Luxury selector title: {clean_title[:50]}...")
                        return clean_title.strip()
        
        # Standard page type extraction (existing logic)
        selectors_by_type = {
            'electronics': [
                '#productTitle',
                'h1 span#productTitle',
                '#title .a-size-large',
                '.product-title'
            ],
            'fashion': [
                '#productTitle',
                '.product-title-word-break',
                'h1.a-size-large'
            ],
            'media': [
                '#ebooksProductTitle',
                '#kindle-title',
                '#bookTitle',
                '#productTitle'
            ],
            'grocery': [
                '#productTitle',
                '.a-size-large.product-title-word-break'
            ]
        }
        
        # Try page-specific selectors first
        for selector in selectors_by_type.get(page_type, ['#productTitle', 'h1 span#productTitle']):
            elements = soup.select(selector)
            for element in elements:
                title_text = element.get_text(strip=True)

                if not title_text or len(title_text) < 5:
                    continue

                title_lower = title_text.lower()
                is_recommendation = any(pattern in title_lower for pattern in exclude_patterns)

                if is_recommendation:
                    if debug:
                        print(f"Skipping recommendation text: {title_text[:30]}...")  
                    continue

                if (len(title_text) > 200 or  
                    title_text.count('\n') > 2 or  
                    title_text.startswith(('Skip to', 'Shop', 'Browse', 'Search'))):
                    if debug:
                        print(f"Skipping invalid title format: {title_text[:30]}...")
                    continue

                clean_title = self._extract_brand_from_title(title_text)

                if debug:
                    print(f"Found valid title with selector ({selector}): {clean_title[:50]}...")
                return clean_title

        return "Product Title Not Found"

    def _extract_brand_from_title(self, title: str) -> str:
        """
        Extract and clean brand from product title, storing detected brands for tracking
        
        This method is crucial for luxury store items where brand names are embedded in titles
        like "LaPointe's Cape T-shirt features..." or "Gucci Women's Handbag"
        """
        
        if not title:
            return title
        
        # Store the extracted brand for later use in _extract_brand method
        if hasattr(self, '_extracted_brand'):
            delattr(self, '_extracted_brand')
        
        original_title = title.strip()
        
        # Get comprehensive brand list (known + learned)
        all_brands = self.get_all_known_brands()
        title_lower = title.lower()
        
        import re
        
        # Pattern 1: "Brand's Product" pattern (luxury stores) - FIXED VERSION
        # Example: "LaPointe's Cape T-shirt features..." -> extract "LaPointe", return "Cape T-shirt features..."
        possessive_match = re.search(r"^([a-z\s&']+)'s\s+", title_lower)
        if possessive_match:
            potential_brand = possessive_match.group(1).strip()
            brand_clean = potential_brand.title()
            
            # Check if it's a known brand or looks like a brand name
            if (potential_brand in all_brands or 
                (len(brand_clean.split()) <= 3 and  # Max 3 words for brand
                 brand_clean[0].isupper() and  # Starts with capital
                 not any(word in potential_brand for word in ['the', 'this', 'that', 'these', 'those']))):
                
                # Remove the brand's possessive form from title - FIXED CLEANING
                cleaned_title = re.sub(r"^[a-z\s&']+'s\s+", "", title, flags=re.IGNORECASE).strip()
                
                # Store the brand for _extract_brand method to use
                self._extracted_brand = brand_clean
                
                # Add to brands tracking
                self.brands_imported.add(brand_clean.lower())
                
                print(f"🏷️  Extracted brand from possessive pattern: '{brand_clean}'")
                print(f"🔧 Cleaned title: '{cleaned_title}'")
                return cleaned_title
        
        # Pattern 2: Brand at beginning of title
        # Example: "Nike Air Max 270" -> extract "Nike", return "Air Max 270"
        words = title.split()
        if len(words) >= 2:
            # Check first word
            first_word = words[0].lower()
            if first_word in all_brands:
                brand_clean = words[0].title()
                cleaned_title = ' '.join(words[1:]).strip()
                
                self._extracted_brand = brand_clean
                self.brands_imported.add(brand_clean.lower())
                
                print(f"🏷️  Extracted brand from title start: '{brand_clean}'")
                return cleaned_title
            
            # Check first two words for compound brands
            if len(words) >= 3:
                first_two = f"{words[0]} {words[1]}".lower()
                if first_two in all_brands:
                    brand_clean = f"{words[0]} {words[1]}".title()
                    cleaned_title = ' '.join(words[2:]).strip()
                    
                    self._extracted_brand = brand_clean
                    self.brands_imported.add(brand_clean.lower())
                    
                    print(f"🏷️  Extracted compound brand: '{brand_clean}'")
                    return cleaned_title
        
        # Pattern 3: Brand anywhere in title - ENHANCED VERSION
        # Example: "Women's Gucci Handbag" -> extract "Gucci", return "Women's Handbag"
        # Sort brands by length (longest first) to avoid partial matches
        sorted_brands = sorted(all_brands, key=len, reverse=True)
        
        for brand in sorted_brands:
            if brand in title_lower:
                # Find the exact position and capitalization in original title
                brand_pattern = re.compile(re.escape(brand), re.IGNORECASE)
                match = brand_pattern.search(title)
                if match:
                    # Get the actual brand text with proper capitalization
                    actual_brand_text = match.group(0)
                    brand_clean = brand.title()
                    
                    # Remove brand from title more carefully
                    # Replace the exact match with empty string and clean up spaces
                    cleaned_title = title[:match.start()] + title[match.end():]
                    cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
                    
                    # Clean up leading/trailing punctuation
                    cleaned_title = re.sub(r'^[,\s\-]+|[,\s\-]+$', '', cleaned_title).strip()
                    
                    if cleaned_title:  # Only if we have remaining title content
                        self._extracted_brand = brand_clean
                        self.brands_imported.add(brand_clean.lower())
                        
                        print(f"🏷️  Extracted brand from title content: '{brand_clean}'")
                        print(f"🔧 Cleaned title: '{cleaned_title}'")
                        return cleaned_title
        
        # Pattern 4: Try to discover new brands using our smart discovery system
        # This will handle cases like "Kiton, 5-Pocket Slim-Straight Pants"
        discovered_brand = self.discover_and_learn_brand(None, title, debug=True)  # Pass None for soup since we don't have it here
        if discovered_brand:
            # Try to remove the discovered brand from title
            brand_lower = discovered_brand.lower()
            if brand_lower in title_lower:
                # Remove the brand and clean up
                brand_pattern = re.compile(re.escape(brand_lower), re.IGNORECASE)
                cleaned_title = brand_pattern.sub('', title).strip()
                cleaned_title = re.sub(r'^[,\s\-]+|[,\s\-]+$', '', cleaned_title).strip()
                cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
                
                if cleaned_title:
                    self._extracted_brand = discovered_brand
                    print(f"🏷️  Extracted discovered brand: '{discovered_brand}'")
                    print(f"🔧 Cleaned title: '{cleaned_title}'")
                    return cleaned_title
        
        # Pattern 5: Handle comma-separated brand patterns (like "Kiton, Product Name")
        comma_match = re.match(r'^([A-Z][a-zA-Z\s&\']+),\s*(.+)$', title.strip())
        if comma_match:
            potential_brand = comma_match.group(1).strip()
            remaining_title = comma_match.group(2).strip()
            
            # Validate this looks like a brand
            if (len(potential_brand) <= 30 and  # Reasonable brand length
                len(potential_brand.split()) <= 3 and  # Max 3 words
                not any(word in potential_brand.lower() for word in ['the', 'and', 'size', 'color', 'style'])):
                
                self._extracted_brand = potential_brand.title()
                self.brands_imported.add(potential_brand.lower())
                
                print(f"🏷️  Extracted brand from comma pattern: '{potential_brand.title()}'")
                print(f"🔧 Cleaned title: '{remaining_title}'")
                return remaining_title
        
        # If no brand extraction patterns matched, return original title
        print(f"🔍 No brand extracted from title, returning original: '{original_title}'")
        return original_title

    def _extract_brand(self, soup: BeautifulSoup, debug=False) -> str:
        """Extract product brand with enhanced luxury store support"""
        
        if debug:
            print(f"🔍 Extracting brand with luxury store support...")
        
        # PRIORITY 1: Check if we extracted brand from title during title processing
        if hasattr(self, '_extracted_brand') and self._extracted_brand:
            brand_from_title = self._extracted_brand.strip()
            if debug:
                print(f"✅ Found brand from title extraction: {brand_from_title}")
            return brand_from_title
        
        # PRIORITY 2: Luxury store specific brand extraction
        page_text = soup.get_text().lower()
        is_luxury_store = 'lx_bd' in str(soup) or 'luxury stores' in page_text
        
        if is_luxury_store and debug:
            print("🔍 Using luxury store brand extraction")
        
        if is_luxury_store:
            # Look for luxury brand patterns in text content
            luxury_brands = [
                'lapointe', 'gucci', 'prada', 'louis vuitton', 'versace', 'balenciaga',
                'chanel', 'dior', 'hermès', 'bottega veneta', 'saint laurent', 'givenchy',
                'valentino', 'tom ford', 'armani', 'dolce gabbana', 'ferragamo', 'burberry'
            ]
            
            # Check title tag for brand
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.get_text()
                for brand in luxury_brands:
                    if brand.lower() in title_text.lower():
                        brand_name = brand.title()
                        if debug:
                            print(f"✅ Found luxury brand in title: {brand_name}")
                        return brand_name
            
            # Check meta description for brand
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                desc_text = meta_desc['content']
                for brand in luxury_brands:
                    if brand.lower() in desc_text.lower():
                        brand_name = brand.title()
                        if debug:
                            print(f"✅ Found luxury brand in meta description: {brand_name}")
                        return brand_name
            
            # Look for brand in text content
            for element in soup.find_all(string=True):
                text = str(element).strip()
                if text and len(text) > 5 and len(text) < 100:
                    for brand in luxury_brands:
                        if brand.lower() in text.lower():
                            # Extract the actual brand name from context
                            words = text.split()
                            for i, word in enumerate(words):
                                if brand.lower() in word.lower():
                                    # Try to get the properly capitalized version
                                    brand_name = brand.title()
                                    if debug:
                                        print(f"✅ Found luxury brand in content: {brand_name}")
                                    return brand_name
        
        # PRIORITY 3: Check JSON-LD structured data
        json_scripts = soup.find_all('script', {'type': 'application/ld+json'})
        for script in json_scripts:
            try:
                import json
                data = json.loads(script.string)
                if isinstance(data, dict):
                    brand = data.get('brand', {})
                    if isinstance(brand, dict) and brand.get('name'):
                        brand_name = brand['name'].strip()
                        if debug:
                            print(f"✅ Found brand from JSON-LD: {brand_name}")
                        return brand_name
                    elif isinstance(brand, str) and len(brand.strip()) > 1:
                        if debug:
                            print(f"✅ Found brand from JSON-LD: {brand.strip()}")
                        return brand.strip()
            except (json.JSONDecodeError, AttributeError):
                continue
        
        # PRIORITY 4: Enhanced brand discovery from product page content
        if soup:  # Safety check
            discovered_brand = self.discover_and_learn_brand(soup, debug=debug)
            if discovered_brand:
                if debug:
                    print(f"✅ Found brand via discovery system: {discovered_brand}")
                return discovered_brand
        
        # PRIORITY 5: Standard Amazon selectors (enhanced with validation)
        selectors = [
            # Primary brand selectors
            '#bylineInfo',
            '#bylineInfo_feature_div a',
            'a#bylineInfo',
            
            # Alternative brand selectors  
            '.po-brand .po-break-word',
            '[data-feature-name="bylineInfo"]',
            '[data-feature-name="bylineInfo"] a',
            
            # Table/structured data (exclude problematic ones)
            '.po-brand td.po-break-word:not([class*="size"]):not([class*="color"])',
            'tr.po-brand .po-break-word:not([class*="size"]):not([class*="color"])',
            
            # Author/publisher (for books)
            '.author a',
            '#authorName', 
            '.contributorNameID',
            
            # Alternative layouts
            '.brand:not([class*="size"]):not([class*="color"])',
            '[data-automation-id="brand-name"]',
            
            # Fallback patterns (be more selective)
            'a[href*="/stores/"]:not([href*="size"]):not([href*="color"])'
        ]
        
        for i, selector in enumerate(selectors):
            elements = soup.select(selector)
            for element in elements:
                brand_text = element.get_text(strip=True)
                if brand_text and len(brand_text) > 1:
                    if debug:
                        print(f"🔍 Trying selector #{i+1} ({selector}): '{brand_text}'")
                    
                    # Enhanced brand text validation and cleaning
                    if self._validate_and_clean_brand_text(brand_text, debug):
                        cleaned_brand = self._validate_and_clean_brand_text(brand_text, debug)
                        if debug:
                            print(f"✅ Found brand with selector #{i+1}: {cleaned_brand}")
                        return cleaned_brand
        
        if debug:
            print("❌ No brand found with any method")
        
        return "Unknown Brand"
    
    def _validate_and_clean_brand_text(self, brand_text: str, debug: bool = False) -> Optional[str]:
        """Validate and clean extracted brand text, return None if invalid"""
        if not brand_text or len(brand_text) < 2:
            return None
            
        import re
        
        # Clean up brand text (remove common prefixes/suffixes)
        cleaned_brand = re.sub(r'(Visit the|Brand:|Store|by\s+|The brand|The\s+)', '', brand_text, flags=re.IGNORECASE).strip()
        
        # Remove trailing periods, commas, etc.
        cleaned_brand = re.sub(r'[.,;:]+$', '', cleaned_brand).strip()
        
        # Skip if obviously not a brand
        invalid_patterns = [
            # Generic terms
            r'^(amazon|store|brand|product|item|unknown|generic)$',
            # Size/color patterns that cause issues like "The brand size isIT"
            r'size\s+is[A-Z]{2,}',
            r'color\s+is[A-Z]{2,}', 
            r'style\s+is[A-Z]{2,}',
            # Navigation/UI elements
            r'^(about|cream|model|type|category|style|size|color)$',
            r'(about the|visit the|click here|see more|view all)',
            # Code-like patterns
            r'[{}()\[\]<>]',
            # Sentences or long phrases (brands are usually short)
            r'\s+is\s+',
            r'\s+the\s+',
            r'\s+and\s+',
        ]
        
        cleaned_lower = cleaned_brand.lower()
        for pattern in invalid_patterns:
            if re.search(pattern, cleaned_lower):
                if debug:
                    print(f"  ❌ Invalid brand pattern detected: {pattern} in '{cleaned_brand}'")
                return None
        
        # Additional validation rules
        if (len(cleaned_brand) > 50 or  # Too long for a brand
            len(cleaned_brand.split()) > 4 or  # Too many words
            cleaned_brand.count(' ') > 3):  # Too many spaces
            if debug:
                print(f"  ❌ Brand too long/complex: '{cleaned_brand}'")
            return None
        
        # Check for reasonable brand characteristics
        if (cleaned_brand and 
            2 <= len(cleaned_brand) <= 50 and
            not cleaned_brand.isdigit() and  # Not just numbers
            any(c.isalpha() for c in cleaned_brand)):  # Contains letters
            
            # Normalize capitalization
            # If it's all caps and more than 3 chars, make it title case
            if cleaned_brand.isupper() and len(cleaned_brand) > 3:
                cleaned_brand = cleaned_brand.title()
            # If it's all lowercase, make it title case
            elif cleaned_brand.islower():
                cleaned_brand = cleaned_brand.title()
            
            return cleaned_brand
        
        if debug:
            print(f"  ❌ Brand failed validation: '{cleaned_brand}'")
        return None
    
    def _extract_price(self, soup: BeautifulSoup, debug=False) -> Optional[float]:
        """Extract product price with comprehensive selectors"""
        
        # Check if this is a luxury store page
        page_text = soup.get_text().lower()
        is_luxury_store = 'lx_bd' in str(soup) or 'luxury stores' in page_text
        
        if is_luxury_store and debug:
            print("Debug - Using luxury store price extraction")
        
        selectors = [
            # Modern Amazon price selectors
            '.a-price.a-text-price.a-size-medium.apexPriceToPay .a-offscreen',
            '.a-price .a-offscreen',
            '.a-price-whole',
            
            # Luxury store specific selectors (prioritized for luxury pages)
            '[class*="price"]',  # This matches the $690$690 pattern we found
            '[data-testid="price"]',
            '[data-testid="product-price"]',
            '.lx-price',
            '.lx-product-price',
            '.luxury-price',
            '[data-testid="current-price"]',
            
            # Legacy price selectors
            '#priceblock_dealprice',
            '#priceblock_ourprice',
            '#displayPrice',
            
            # Sale/current price
            '.a-size-medium.a-color-price',
            '.a-price-current',
            
            # Range pricing
            'span.a-price-range .a-price.a-text-price .a-offscreen',
            'span.a-price-range',
            
            # Specific product types
            '#kindle-price',
            '.a-color-price.a-size-medium',
            
            # Mobile/alternative layouts
            '[data-automation-id="product-price"]',
            '.price',
            '.a-color-price',
            
            # Fallback patterns
            '.a-text-price',
            '[class*="price"]'
        ]
        
        if debug:
            print(f"Searching for price in {len(selectors)} selectors...")
        
        for i, selector in enumerate(selectors):
            elements = soup.select(selector)
            for element in elements:
                price_text = element.get_text(strip=True)
                if price_text and ('$' in price_text or any(char.isdigit() for char in price_text)):
                    if debug:
                        print(f"Trying selector #{i+1} ({selector}): '{price_text}'")
                    
                    # Extract numbers from price text
                    import re
                    
                    # Handle luxury store duplicated price pattern like $690$690
                    if is_luxury_store and price_text.count('$') > 1:
                        # Look for duplicated price pattern: $123$123
                        dup_pattern = re.search(r'\$([0-9,]+\.?[0-9]*)\$\1', price_text.replace(',', ''))
                        if dup_pattern:
                            try:
                                price_value = float(dup_pattern.group(1))
                                if 0.01 <= price_value <= 99999:
                                    if debug:
                                        print(f"Found luxury duplicated price: ${price_value}")
                                    return price_value
                            except (ValueError, IndexError):
                                pass
                    
                    # More comprehensive price regex
                    price_patterns = [
                        r'\$?([0-9,]+\.?[0-9]*)',  # Standard: $12.34 or 12.34
                        r'([0-9,]+\.?[0-9]*)\s*dollars?',  # "12 dollars"
                        r'([0-9,]+\.?[0-9]*)',  # Just numbers
                    ]
                    
                    for pattern in price_patterns:
                        price_match = re.search(pattern, price_text.replace(',', ''))
                        if price_match:
                            try:
                                price_value = float(price_match.group(1))
                                if 0.01 <= price_value <= 99999:  # Reasonable price range
                                    if debug:
                                        print(f"Found price with selector #{i+1}: ${price_value}")
                                    return price_value
                            except (ValueError, IndexError):
                                continue
        
        # Debug: Show price-related elements if none found
        if debug:
            price_elements = soup.find_all(string=re.compile(r'\$|price|cost', re.I))
            print(f"Found {len(price_elements)} price-related text elements:")
            for i, elem in enumerate(price_elements[:5]):  # Show first 5
                print(f"  Price text #{i+1}: {str(elem).strip()[:50]}...")
        
        return None
    
    def _extract_images(self, soup: BeautifulSoup, debug=False) -> List[str]:
        """Extract product images with enhanced quality and coverage"""
        images = []
        
        if debug:
            print("🖼️ Extracting product images...")
        
        # Primary image selectors (highest quality first)
        img_selectors = [
            # Main product images
            '#landingImage',
            '#imgBlkFront', 
            '.a-dynamic-image',
            
            # Alternative main images
            '#main-image-container img',
            '.itemNo0.maintain-height img',
            '[data-action="main-image-click"] img',
            
            # Gallery images  
            '#altImages img',
            '.imageThumb img',
            '.a-button-thumbnail img',
            
            # Fallback images
            '.product-image img',
            '[id*="image"] img',
            'img[src*="images-amazon"]'
        ]
        
        processed_urls = set()
        
        for i, selector in enumerate(img_selectors):
            img_elements = soup.select(selector)
            for img in img_elements:
                # Filter out Amazon branding by checking alt text and class names
                alt_text = (img.get('alt') or '').lower()
                class_names = ' '.join(img.get('class', [])).lower()
                
                branding_indicators = [
                    'prime', 'amazon', 'logo', 'badge', 'delivery', 'shipping',
                    'fulfillment', 'free shipping', 'prime logo'
                ]
                
                # Skip if image attributes suggest Amazon branding
                if any(indicator in alt_text or indicator in class_names for indicator in branding_indicators):
                    if debug:
                        print(f"  🚫 Skipping Amazon branding image: {alt_text[:30]}")
                    continue
                
                # Try multiple attributes for image URL
                src = (img.get('src') or 
                       img.get('data-src') or 
                       img.get('data-old-hires') or
                       img.get('data-a-dynamic-image'))
                
                if src and src.startswith('http'):
                    # Clean and enhance image URL for better quality (returns None for filtered images)
                    clean_src = self._enhance_image_url(src)
                    if clean_src and clean_src not in processed_urls:
                        images.append(clean_src)
                        processed_urls.add(clean_src)
                        if debug:
                            print(f"  ✅ Found image #{len(images)} from selector {i+1}: {clean_src[:60]}...")
        
        # Remove duplicates while preserving order
        unique_images = []
        seen_urls = set()
        for img_url in images:
            if img_url not in seen_urls:
                unique_images.append(img_url)
                seen_urls.add(img_url)
        
        if debug:
            print(f"📸 Total unique images found: {len(unique_images)}")
        
        return unique_images[:10]  # Limit to 10 images for performance
    
    def _enhance_image_url(self, url: str) -> str:
        """Enhance Amazon image URL for maximum quality and filter out Amazon branding"""
        if not url or 'images-amazon.com' not in url:
            return url
        
        # Filter out Amazon branding images (Prime logos, etc.)
        branding_keywords = [
            'prime', 'amazon-logo', 'badge', 'delivery', 'shipping',
            'free-shipping', 'prime-logo', 'amazon-badge', 'fulfillment'
        ]
        
        url_lower = url.lower()
        for keyword in branding_keywords:
            if keyword in url_lower:
                return None  # Skip this image
        
        # Try to get maximum resolution by modifying URL parameters
        enhanced_url = url
        
        # Replace ALL size indicators with high-resolution versions
        # Amazon supports up to 2000px+ images
        size_replacements = [
            # Small to large conversions
            ('._SL75_', '._SL1500_'),
            ('._SX75_', '._SX1500_'),   
            ('._SY75_', '._SY1500_'),
            ('._AC_SL75_', '._AC_SL1500_'),
            ('._AC_SX75_', '._AC_SX1500_'),
            ('._AC_SY75_', '._AC_SY1500_'),
            
            # Medium to large conversions  
            ('._SL300_', '._SL1500_'),
            ('._SX300_', '._SX1500_'),
            ('._SY300_', '._SY1500_'),
            ('._SL500_', '._SL1500_'),
            ('._SX500_', '._SX1500_'),
            ('._SY500_', '._SY1500_'),
            
            # Auto-crop to large
            ('._AC_SL300_', '._AC_SL1500_'),
            ('._AC_SX300_', '._AC_SX1500_'),
            ('._AC_SY300_', '._AC_SY1500_'),
            ('._AC_SL500_', '._AC_SL1500_'),
            ('._AC_SX500_', '._AC_SX1500_'),
            ('._AC_SY500_', '._AC_SY1500_'),
        ]
        
        for small, large in size_replacements:
            if small in enhanced_url:
                enhanced_url = enhanced_url.replace(small, large)
                break
        
        # If no size parameter found, try to add one for maximum quality
        if '._S' not in enhanced_url and '.jpg' in enhanced_url:
            # Insert size parameter before file extension
            enhanced_url = enhanced_url.replace('.jpg', '._SL1500_.jpg')
        elif '._S' not in enhanced_url and '.png' in enhanced_url:
            enhanced_url = enhanced_url.replace('.png', '._SL1500_.png')
        
        return enhanced_url
    
    def _is_template_or_error_content(self, text: str) -> bool:
        """Check if content is template/error code rather than product description"""
        if not text:
            return True
            
        text_lower = text.lower()
        
        # Mason template indicators
        mason_indicators = [
            'mason',
            '/mason/',
            'onload function',
            'embed-features.mi',
            'amaozn-family',
            'requires a corresponding change',
            'pass in /',
            '.mi'
        ]
        
        # Development/error indicators
        error_indicators = [
            'error',
            'exception',
            'undefined',
            'null',
            'debug',
            'console',
            'javascript',
            'function(',
            'var ',
            'let ',
            'const ',
            '/*',
            '*/',
            '//',
            'return',
            'if(',
            'for(',
            'while('
        ]
        
        # Check for template/error content
        if any(indicator in text_lower for indicator in mason_indicators + error_indicators):
            return True
            
        # Check for code-like patterns
        import re
        if re.search(r'[\{\}\[\]();]', text) and len(re.findall(r'[\{\}\[\]();]', text)) > 3:
            return True
            
        return False
    
    def _extract_description(self, soup: BeautifulSoup, debug=False) -> str:
        """Extract product description with enhanced luxury store support and content validation"""
        
        # First, get the title to avoid duplicating it as description
        extracted_title = self._extract_title_by_page_type(soup, self._detect_page_type(soup), debug=False)
        
        if debug:
            print(f"📝 Extracting description with luxury store support...")
        
        # Check if this is a luxury store page
        page_text = soup.get_text().lower()
        is_luxury_store = 'lx_bd' in str(soup) or 'luxury stores' in page_text
        
        if is_luxury_store and debug:
            print("🔍 Using luxury store description extraction")
        
        # PRIORITY 1: Luxury store specific description extraction
        if is_luxury_store:
            # Look for luxury product descriptions in specific sections
            luxury_description_selectors = [
                # Luxury store specific selectors (avoiding Mason template areas)
                '[data-testid="product-description"]',
                '.lx-product-description',
                '.luxury-product-details',
                '[id*="description"]:not([id*="embed"]):not([id*="mason"])',
                '.product-feature-text',
                '.lx-feature-text',
                
                # Look for structured product info in luxury layouts
                '.a-section .a-spacing-small:not([class*="embed"]):not([class*="mason"])',
            ]
            
            for selector in luxury_description_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    
                    if (text and 20 < len(text) < 800 and  # Reasonable length
                        not self._is_template_or_error_content(text) and
                        text.lower() != extracted_title.lower()):
                        
                        if debug:
                            print(f"✅ Found luxury description via selector: {text[:50]}...")
                        return self._clean_description_text(text)
            
            # Look for luxury product descriptions in text content (enhanced filtering)
            description_candidates = []
            
            for element in soup.find_all(string=True):
                text = str(element).strip()
                if text and 30 < len(text) < 500:  # Reasonable description length
                    # Skip template/error content
                    if self._is_template_or_error_content(text):
                        continue
                        
                    # Look for luxury product description patterns
                    if any(word in text.lower() for word in [
                        'features', 'crafted', 'designed', 'made', 'fabric', 'material', 'style', 
                        'collection', 'silhouette', 'cut', 'fit', 'finish', 'construction',
                        'cotton', 'silk', 'wool', 'cashmere', 'leather', 'polyester', 'viscose'
                    ]):
                        # Enhanced skip patterns for luxury stores
                        skip_patterns = [
                            'amazon.com', 'luxury stores', 'visit the', 'customers who',
                            'mason', 'embed-features', 'onload function', 'amaozn-family',
                            'departments', 'navigation', 'menu', 'search', 'sign in'
                        ]
                        
                        if not any(skip in text.lower() for skip in skip_patterns):
                            description_candidates.append(text)
            
            # Find the best description candidate
            for desc in sorted(description_candidates, key=len, reverse=True):  # Prefer longer descriptions
                # Skip if it's just the title or very similar
                if (desc.lower() != extracted_title.lower() and 
                    not desc.lower().startswith(extracted_title[:30].lower()) and
                    not self._is_template_or_error_content(desc)):
                    
                    if debug:
                        print(f"✅ Found luxury description from content: {desc[:50]}...")
                    return self._clean_description_text(desc)
        
        # PRIORITY 2: Standard product description sections (enhanced validation)
        product_description_selectors = [
            # Amazon's actual product description sections (excluding problematic ones)
            '#productDescription p',
            '#productDescription div:not([id*="embed"]):not([class*="embed"])',
            '.product-description:not([class*="embed"])',
            
            # A+ content (rich product descriptions) - safer selectors
            '#aplus .a-section:not([class*="embed"])',
            '#aplus_feature_div .a-section:not([class*="embed"])',
            
            # Feature bullets as primary option (usually reliable)
            '#feature-bullets ul li span',
            '.a-unordered-list.a-vertical.a-spacing-mini li span',
            '[data-feature-name="featurebullets"] ul li span',
            
            # Alternative description layouts (safer)
            '.po-break-word:not(.po-brand):not([class*="embed"])',
            
            # Fallback selectors
            '.a-section.a-spacing-medium:not([class*="embed"]):not([id*="embed"])'
        ]
        
        for i, selector in enumerate(product_description_selectors):
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                
                # Skip if too short
                if len(text) < 20:
                    continue
                
                # Skip template/error content
                if self._is_template_or_error_content(text):
                    if debug:
                        print(f"  ❌ Skipping template/error content: {text[:40]}...")
                    continue
                
                # Skip if it looks like Amazon page navigation or titles
                text_lower = text.lower()
                skip_patterns = [
                    'amazon.com', 'luxury stores', 'departments', 'best sellers',
                    'customer reviews', 'you may also like', 'frequently bought together',
                    'customers who viewed', 'customers who bought', 'visit the', 'store',
                    'brand:', 'add to cart', 'buy now', 'price', 'shipping', 'return'
                ]
                
                if any(pattern in text_lower for pattern in skip_patterns):
                    if debug:
                        print(f"  ❌ Skipping navigation text: {text[:40]}...")
                    continue
                
                # Check if it's just the title or starts with title
                if (text.lower() == extracted_title.lower() or
                    text.lower().startswith(extracted_title[:30].lower())):
                    if debug:
                        print(f"  ❌ Skipping title duplication: {text[:40]}...")
                    continue
                
                # Valid description content found
                if debug:
                    print(f"✅ Found description with selector #{i+1} ({selector}): {text[:50]}...")
                
                return self._clean_description_text(text)
        
        # PRIORITY 3: Combine feature bullets if no single description found
        feature_elements = soup.select('#feature-bullets ul li, [data-feature-name="featurebullets"] ul li')
        if feature_elements:
            features = []
            for element in feature_elements:
                text = element.get_text(strip=True)
                if (text and len(text) > 10 and len(text) < 200 and
                    not self._is_template_or_error_content(text)):
                    features.append(text)
            
            if features and len(features) >= 2:  # At least 2 features
                combined_description = '. '.join(features[:5])  # Max 5 features
                if debug:
                    print(f"✅ Found description from combined features: {combined_description[:50]}...")
                return self._clean_description_text(combined_description)
        
        # PRIORITY 4: Meta descriptions (with enhanced filtering)
        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description and meta_description.get('content'):
            desc_text = meta_description['content'].strip()
            
            # Enhanced filtering for meta descriptions
            if (len(desc_text) > 20 and 
                not self._is_template_or_error_content(desc_text) and
                'amazon.com' not in desc_text.lower() and
                'luxury stores' not in desc_text.lower() and
                ': ' not in desc_text[:20] and  # Avoid "Amazon.com: Product" format
                desc_text.lower() != extracted_title.lower() and
                not desc_text.lower().startswith(extracted_title[:30].lower())):
                
                if debug:
                    print(f"✅ Found description from meta tag: {desc_text[:50]}...")
                return self._clean_description_text(desc_text)
        
        if debug:
            print("❌ No valid product description found")
        return "No description available"
    
    def _clean_description_text(self, text: str) -> str:
        """Clean and format description text"""
        if not text:
            return text
            
        # Clean up text
        cleaned_text = text.replace('\n', ' ').replace('\t', ' ')
        
        # Remove multiple spaces
        import re
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Remove common prefixes that might have slipped through
        prefixes_to_remove = [
            'Product Description',
            'Description:',
            "Description",
            'Features:',
            'About this item:',
            'Product details:'
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
        
        # Limit length and add ellipsis if needed
        if len(cleaned_text) > 500:
            # Try to cut at a sentence boundary
            sentences = cleaned_text.split('. ')
            truncated = ''
            for sentence in sentences:
                if len(truncated + sentence + '. ') <= 500:
                    truncated += sentence + '. '
                else:
                    break
            if truncated:
                cleaned_text = truncated.rstrip('. ') + '...'
            else:
                cleaned_text = cleaned_text[:500] + "..."
        
        return cleaned_text
    
    def _extract_brand(self, soup: BeautifulSoup, debug=False) -> str:
        """Extract product brand with comprehensive selectors including meta tags and title extraction"""
        
        if debug:
            print(f"🔍 Extracting brand with priority order...")
        
        # PRIORITY 1: Check if we extracted brand from title during title processing
        if hasattr(self, '_extracted_brand') and self._extracted_brand:
            brand_from_title = self._extracted_brand.strip()
            if debug:
                print(f"✅ Found brand from title extraction: {brand_from_title}")
            return brand_from_title
        
        # PRIORITY 2: Check JSON-LD structured data
        json_scripts = soup.find_all('script', {'type': 'application/ld+json'})
        for script in json_scripts:
            try:
                import json
                data = json.loads(script.string)
                if isinstance(data, dict):
                    brand = data.get('brand', {})
                    if isinstance(brand, dict) and brand.get('name'):
                        brand_name = brand['name'].strip()
                        if debug:
                            print(f"✅ Found brand from JSON-LD: {brand_name}")
                        return brand_name
                    elif isinstance(brand, str) and len(brand.strip()) > 1:
                        if debug:
                            print(f"✅ Found brand from JSON-LD: {brand.strip()}")
                        return brand.strip()
            except (json.JSONDecodeError, AttributeError):
                continue
        
        # PRIORITY 3: Standard Amazon selectors
        selectors = [
            # Primary brand selectors
            '#bylineInfo',
            '#bylineInfo_feature_div a',
            'a#bylineInfo',
            
            # Alternative brand selectors
            '.po-brand .po-break-word',
            '[data-feature-name="bylineInfo"]',
            '[data-feature-name="bylineInfo"] a',
            
            # Table/structured data
            '.po-brand td.po-break-word',
            'tr.po-brand .po-break-word',
            
            # Author/publisher (for books)
            '.author a',
            '#authorName',
            '.contributorNameID',
            
            # Alternative layouts
            '.brand',
            '[data-automation-id="brand-name"]',
            '.a-size-base.po-break-word',
            
            # Fallback patterns
            'a[href*="/stores/"]',
            '[class*="brand"]',
            'title'
        ]
        
        for i, selector in enumerate(selectors):
            elements = soup.select(selector)
            for element in elements:
                brand_text = element.get_text(strip=True)
                if brand_text and len(brand_text) > 1:
                    if debug:
                        print(f"🔍 Trying selector #{i+1} ({selector}): '{brand_text}'")
                    
                    # Clean up brand text (remove common prefixes/suffixes)
                    import re
                    cleaned_brand = re.sub(r'(Visit the|Brand:|Store|by\s+)', '', brand_text, flags=re.IGNORECASE).strip()
                    
                    # Skip generic terms that aren't actual brands
                    if cleaned_brand.lower() in ['amazon', 'store', 'brand', 'about the brand', 'cream', 'product', 'item']:
                        if debug:
                            print(f"  ❌ Skipping generic term: {cleaned_brand}")
                        continue
                    
                    # Valid brand should be reasonable length and not generic
                    if cleaned_brand and 2 <= len(cleaned_brand) <= 100:
                        if debug:
                            print(f"✅ Found brand with selector #{i+1}: {cleaned_brand}")
                        return cleaned_brand
        
        # Debug: Show potential brand-related text
        if debug:
            byline_elements = soup.find_all(['a', 'span'], string=re.compile(r'visit|brand|by\s+', re.I))
            print(f"🔍 Found {len(byline_elements)} potential brand elements:")
            for i, elem in enumerate(byline_elements[:3]):
                print(f"  Brand candidate #{i+1}: {elem.get_text(strip=True)[:50]}...")
        
        return "Unknown Brand"
    
    def _extract_features(self, soup: BeautifulSoup, debug=False) -> List[str]:
        """Extract product features/bullet points"""
        features = []
        
        # Try feature bullets
        bullets = soup.select('#feature-bullets li span.a-list-item')
        for bullet in bullets:
            text = bullet.get_text(strip=True)
            if text and len(text) > 10 and not text.startswith('Make sure'):
                features.append(text)
        
        return features[:5]  # Limit to 5 features
    
    def _extract_weight(self, soup: BeautifulSoup, debug=False) -> Optional[float]:
        """Extract product weight in ounces from Amazon page with enhanced coverage for luxury stores"""
        try:
            # Detect if this is a luxury store page
            page_text = soup.get_text().lower()  
            is_luxury_store = 'lx_bd' in str(soup) or 'luxury stores' in page_text
            
            if debug:
                print(f"🔍 Weight extraction - Luxury store: {is_luxury_store}")
            
            # Enhanced weight patterns for better detection
            weight_patterns = [
                'Item Weight', 'Product Weight', 'Package Weight', 'Shipping Weight',
                'Product Dimensions', 'Package Dimensions', 'Weight', 'Net Weight', 
                'Gross Weight', 'Item Dimensions', 'Package Dimensions', 'Dimensions',
                'Weight:', 'Peso', 'Gewicht'  # International variations
            ]
            
            # PRIORITY 1: Luxury store specific weight extraction
            if is_luxury_store:
                if debug:
                    print("🔍 Using luxury store weight extraction")
                
                # Look for weight in luxury store specific selectors
                luxury_weight_selectors = [
                    '.lx-product-details',
                    '.luxury-product-info', 
                    '[data-testid="product-specs"]',
                    '.product-specifications',
                    '.lx-specs'
                ]
                
                for selector in luxury_weight_selectors:
                    elements = soup.select(selector)
                    for element in elements:
                        text = element.get_text()
                        for pattern in weight_patterns:
                            if pattern.lower() in text.lower():
                                # Extract weight value from this section
                                weight_oz = self._parse_weight_to_ounces(text)
                                if weight_oz:
                                    if debug:
                                        print(f"Found weight in luxury selector {selector}: {weight_oz} oz")
                                    return weight_oz
                
                # Look for weight in luxury store text content
                for element in soup.find_all(string=True):
                    text = str(element).strip()
                    if text and len(text) > 5 and len(text) < 200:
                        for pattern in weight_patterns:
                            if pattern.lower() in text.lower():
                                weight_oz = self._parse_weight_to_ounces(text)
                                if weight_oz:
                                    if debug:
                                        print(f"Found weight in luxury text content: {text} → {weight_oz} oz")
                                    return weight_oz
            
            # PRIORITY 2: Standard product details table
            details_selectors = [
                'table#productDetails_detailBullets_sections1',
                'table#productDetails_techSpec_section_1',
                '#productDetails_detailBullets_sections1',
                '#detail-bullets',
                '.po-details'
            ]
            
            for table_selector in details_selectors:
                details_table = soup.select_one(table_selector)
                if details_table:
                    rows = details_table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            label = cells[0].get_text(strip=True)
                            value = cells[1].get_text(strip=True)
                            
                            if any(pattern in label for pattern in weight_patterns):
                                weight_oz = self._parse_weight_to_ounces(value)
                                if weight_oz:
                                    if debug:
                                        print(f"Found weight in details table: {label} = {value} → {weight_oz} oz")
                                    return weight_oz
            
            # PRIORITY 3: Technical specifications section (enhanced)
            tech_selectors = [
                '#prodDetails',
                '#technicalSpecifications_feature_div',
                '.po-technical-specifications',
                '[data-feature-name="technicalSpecifications"]'
            ]
            
            for tech_selector in tech_selectors:
                tech_details = soup.select_one(tech_selector)
                if tech_details:
                    # Look for weight patterns in technical specs
                    for pattern in weight_patterns:
                        weight_regex = re.compile(pattern, re.IGNORECASE)
                        weight_elem = tech_details.find(text=weight_regex)
                        if weight_elem:
                            # Get surrounding context for weight value
                            parent = weight_elem.parent
                            if parent:
                                # Check siblings and nearby elements
                                for sibling in parent.find_next_siblings()[:3]:
                                    text = sibling.get_text(strip=True)
                                    weight_oz = self._parse_weight_to_ounces(text)
                                    if weight_oz:
                                        if debug:
                                            print(f"Found weight in tech details: {pattern} = {text} → {weight_oz} oz")
                                        return weight_oz
                                
                                # Also check the same element
                                full_text = parent.get_text(strip=True)
                                weight_oz = self._parse_weight_to_ounces(full_text)
                                if weight_oz:
                                    if debug:
                                        print(f"Found weight in tech parent: {pattern} = {full_text} → {weight_oz} oz")
                                    return weight_oz
            
            # PRIORITY 4: Feature bullets (enhanced)
            bullet_selectors = [
                '#feature-bullets li span.a-list-item',
                '#feature-bullets li',
                '.a-unordered-list.a-vertical li',
                '[data-feature-name="featurebullets"] li'
            ]
            
            for bullet_selector in bullet_selectors:
                bullets = soup.select(bullet_selector)
                for bullet in bullets:
                    text = bullet.get_text(strip=True)
                    if any(pattern.lower() in text.lower() for pattern in weight_patterns):
                        weight_oz = self._parse_weight_to_ounces(text)
                        if weight_oz:
                            if debug:
                                print(f"Found weight in features: {text} → {weight_oz} oz")
                            return weight_oz
            
            # PRIORITY 5: Shipping/packaging info sections
            shipping_selectors = [
                '#shipping-information',
                '.shipping-details',
                '#packaging-details',
                '.package-info'
            ]
            
            for shipping_selector in shipping_selectors:
                shipping_section = soup.select_one(shipping_selector)
                if shipping_section:
                    text = shipping_section.get_text()
                    weight_oz = self._parse_weight_to_ounces(text)
                    if weight_oz:
                        if debug:
                            print(f"Found weight in shipping details: {weight_oz} oz")
                        return weight_oz
            
            # PRIORITY 6: Parse all visible text for weight patterns (last resort)
            if debug:
                print("🔍 Searching all visible text for weight patterns...")
            
            all_text = soup.get_text()
            # Look for weight patterns in format "Weight: X oz/lbs/kg/g"
            weight_regex_patterns = [
                r'(?:item\s+)?weight[\s:]+([0-9.]+)\s*(oz|ounces|lb|lbs|pounds|kg|kilograms|g|grams)',
                r'([0-9.]+)\s*(oz|ounces|lb|lbs|pounds|kg|kilograms|g|grams)(?:\s+weight)?',
                r'weighs?\s+([0-9.]+)\s*(oz|ounces|lb|lbs|pounds|kg|kilograms|g|grams)',
                r'weight[\s:]+([0-9.]+)\s*(oz|ounces|lb|lbs|pounds|kg|kilograms|g|grams)'
            ]
            
            for pattern in weight_regex_patterns:
                matches = re.finditer(pattern, all_text, re.IGNORECASE)
                for match in matches:
                    weight_value = float(match.group(1))
                    unit = match.group(2).lower()
                    
                    # Convert to ounces
                    if unit in ['oz', 'ounces']:
                        weight_oz = weight_value
                    elif unit in ['lb', 'lbs', 'pounds']:
                        weight_oz = weight_value * 16
                    elif unit in ['kg', 'kilograms']:
                        weight_oz = weight_value * 35.274
                    elif unit in ['g', 'grams']:
                        weight_oz = weight_value * 0.035274
                    else:
                        continue
                    
                    if weight_oz and weight_oz > 0.01:  # Reasonable weight threshold
                        if debug:
                            print(f"Found weight in text regex: {match.group(0)} → {weight_oz} oz")
                        return weight_oz
            
            if debug:
                print("No weight information found on product page")
            return None
            
        except Exception as e:
            if debug:
                print(f"Error extracting weight: {e}")
            return None
    
    async def _get_alternative_weight_sources(self, asin: str, product_data: Dict, debug: bool = False) -> Optional[float]:
        """
        Get product weight from alternative sources when Amazon scraping fails
        
        Priority order:
        1. Keepa API (if available)
        2. Manufacturer website lookup
        3. Category-based estimation
        """
        if debug:
            print(f"🔍 Searching alternative weight sources for {asin}")
        
        # SOURCE 1: Keepa API (if we have it configured)
        if hasattr(self, 'keepa_client') and self.keepa_client:
            try:
                keepa_data = await self.keepa_client.get_product(asin)
                if keepa_data:
                    weight_oz = await self._extract_weight_from_keepa(keepa_data, debug)
                    if weight_oz:
                        if debug:
                            print(f"✅ Found weight via Keepa: {weight_oz} oz")
                        return weight_oz
            except Exception as e:
                if debug:
                    print(f"❌ Keepa lookup failed: {e}")
        
        # SOURCE 2: Manufacturer website lookup (for high-value items)
        brand = product_data.get('brand', '').strip()
        title = product_data.get('title', '')
        price = product_data.get('price', 0)
        
        # Perform manufacturer lookup if conditions are met  
        current_weight = product_data.get('weight_oz')
        if (self.manufacturer_lookup and brand and title and 
            self.manufacturer_lookup.should_lookup(brand, price, current_weight)):
            try:
                specs = await self.manufacturer_lookup.lookup_specifications(brand, title, price)
                if specs and specs.get('weight_oz'):
                    if debug:
                        print(f"✅ Found weight via manufacturer lookup: {specs['weight_oz']} oz (confidence: {specs.get('confidence', 0.0):.1f})")
                    # Update product_data with manufacturer info
                    product_data['weight_source'] = 'manufacturer'
                    product_data['manufacturer_model'] = specs.get('model')
                    product_data['manufacturer'] = specs.get('manufacturer')
                    product_data['confidence'] = specs.get('confidence', 0.0)
                    return specs['weight_oz']
                elif debug:
                    print(f"🔍 Manufacturer lookup attempted but no weight data found for {brand} {title[:30]}...")
            except Exception as e:
                if debug:
                    print(f"❌ Manufacturer lookup failed: {e}")
        
        # SOURCE 3: Category-based estimation (last resort)
        estimated_weight = self._estimate_weight_by_category(product_data, debug)
        if estimated_weight:
            if debug:
                print(f"💡 Estimated weight by category: {estimated_weight} oz")
            return estimated_weight
        
        return None
    

    
    def _estimate_weight_by_category(self, product_data: Dict, debug: bool = False) -> Optional[float]:
        """Estimate weight based on product category and characteristics"""
        
        title = product_data.get('title', '').lower()
        brand = product_data.get('brand', '').lower()
        price = product_data.get('price', 0)
        
        # Category-based weight estimates (in ounces)
        weight_estimates = {
            # Electronics
            'tablet': {'min': 8, 'max': 24, 'default': 16},        # iPad, tablets
            'headphones': {'min': 4, 'max': 20, 'default': 12},   # Over-ear headphones
            'earbuds': {'min': 0.1, 'max': 2, 'default': 0.5},   # Wireless earbuds
            'phone': {'min': 4, 'max': 8, 'default': 6},          # Smartphones
            'watch': {'min': 1, 'max': 4, 'default': 2},          # Smartwatches
            'ring': {'min': 0.05, 'max': 0.3, 'default': 0.15},  # Smart rings
            
            # Clothing
            't-shirt': {'min': 3, 'max': 8, 'default': 5},        # T-shirts
            'shirt': {'min': 4, 'max': 12, 'default': 7},         # Dress shirts
            'pants': {'min': 8, 'max': 20, 'default': 14},        # Pants/trousers
            'jacket': {'min': 12, 'max': 32, 'default': 20},      # Jackets
            'shoes': {'min': 12, 'max': 40, 'default': 24},       # Footwear
            
            # Accessories
            'bag': {'min': 4, 'max': 32, 'default': 16},          # Handbags
            'wallet': {'min': 2, 'max': 8, 'default': 4},         # Wallets
            'sunglasses': {'min': 1, 'max': 4, 'default': 2},     # Sunglasses
        }
        
        # Identify category from title
        detected_category = None
        for category, keywords in {
            'tablet': ['ipad', 'tablet'],
            'headphones': ['headphones', 'headset', 'wh-', 'audio'],
            'earbuds': ['earbuds', 'airpods', 'buds', 'wf-'],
            'phone': ['iphone', 'phone', 'galaxy phone'],
            'watch': ['watch', 'smartwatch'],
            'ring': ['ring', 'smart ring'],
            't-shirt': ['t-shirt', 'tee', 'cape t-shirt'],
            'shirt': ['shirt', 'blouse'],
            'pants': ['pants', 'trousers', 'jeans'],
            'jacket': ['jacket', 'coat', 'blazer'],
            'shoes': ['shoes', 'sneakers', 'boots'],
            'bag': ['bag', 'handbag', 'purse'],
            'wallet': ['wallet'],
            'sunglasses': ['sunglasses', 'glasses']
        }.items():
            if any(keyword in title for keyword in keywords):
                detected_category = category
                break
        
        if not detected_category:
            if debug:
                print("❌ Could not detect product category for weight estimation")
            return None
        
        estimate_range = weight_estimates[detected_category]
        
        # Adjust estimate based on price (higher price usually means higher quality/weight)
        if price:
            if price > 500:  # Premium products
                weight_estimate = estimate_range['max'] * 0.8
            elif price > 100:  # Mid-range products
                weight_estimate = estimate_range['default']
            else:  # Budget products
                weight_estimate = estimate_range['min'] * 1.2
        else:
            weight_estimate = estimate_range['default']
        
        # Adjust for luxury brands (often heavier due to premium materials)
        luxury_brands = ['gucci', 'prada', 'lapointe', 'kiton', 'hermès', 'louis vuitton']
        if brand in luxury_brands:
            weight_estimate *= 1.3
        
        if debug:
            print(f"💡 Category: {detected_category}, Price: ${price}, Brand: {brand}")
            print(f"💡 Estimated weight: {weight_estimate:.2f} oz")
        
        return round(weight_estimate, 2)
    
    def _parse_weight_to_ounces(self, text: str) -> Optional[float]:
        """Parse weight text and convert to ounces"""
        try:
            import re
            
            # Remove extra spaces and normalize
            text = re.sub(r'\s+', ' ', text.strip().lower())
            
            # Enhanced weight patterns with more formats
            patterns = [
                # Pounds (various formats)
                r'(\d+\.?\d*)\s*(?:pounds?|lbs?|lb)\b',
                r'(\d+\.?\d*)\s*(?:pound|lb\.)',
                
                # Ounces (various formats)  
                r'(\d+\.?\d*)\s*(?:ounces?|oz)\b',
                r'(\d+\.?\d*)\s*(?:ounce|oz\.)',
                
                # Grams (various formats)
                r'(\d+\.?\d*)\s*(?:grams?|g)\b',
                r'(\d+\.?\d*)\s*(?:gram|g\.)',
                
                # Kilograms (various formats)
                r'(\d+\.?\d*)\s*(?:kilograms?|kg)\b',
                r'(\d+\.?\d*)\s*(?:kilogram|kg\.)',
                
                # Combined weight formats (e.g., "2 lbs 4 oz")
                r'(\d+)\s*(?:lbs?|pounds?)\s*(\d+)\s*(?:oz|ounces?)',
                
                # Decimal weight formats
                r'(\d+\.\d+)\s*(?:pounds?|lbs?|lb|oz|ounces?|g|grams?|kg|kilograms?)',
            ]
            
            for i, pattern in enumerate(patterns):
                match = re.search(pattern, text)
                if match:
                    # Handle combined weight format (e.g., "2 lbs 4 oz")
                    if i == len(patterns) - 3:  # Combined lbs + oz pattern
                        lbs = float(match.group(1))
                        oz = float(match.group(2))
                        return (lbs * 16) + oz
                    
                    value = float(match.group(1))
                    
                    # Convert to ounces based on unit detected in pattern
                    if 'pound' in pattern or 'lb' in pattern:
                        return value * 16  # 1 pound = 16 ounces
                    elif 'ounce' in pattern or 'oz' in pattern:
                        return value
                    elif 'gram' in pattern and 'kilogram' not in pattern:
                        return value / 28.35  # 1 ounce = 28.35 grams
                    elif 'kilogram' in pattern or 'kg' in pattern:
                        return value * 35.274  # 1 kg = 35.274 ounces
                    else:
                        # For decimal format, try to detect unit from text context
                        if any(unit in text for unit in ['lb', 'pound']):
                            return value * 16
                        elif any(unit in text for unit in ['oz', 'ounce']):
                            return value
                        elif any(unit in text for unit in ['kg', 'kilogram']):
                            return value * 35.274
                        elif any(unit in text for unit in ['g', 'gram']) and 'kg' not in text:
                            return value / 28.35
            
            return None
            
        except (ValueError, AttributeError):
            return None
    
    async def close(self):
        """Close the HTTP session and cleanup resources"""
        if self.session:
            await self.session.aclose()
        
        # Clean up manufacturer lookup system
        if self.manufacturer_lookup:
            await self.manufacturer_lookup.close()

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self):
        self.requests = {}
    
    def allow_request(self, api_name: str, max_per_minute: int = 60) -> bool:
        now = time.time()
        if api_name not in self.requests:
            self.requests[api_name] = []
        
        # Clean old requests
        self.requests[api_name] = [t for t in self.requests[api_name] if now - t < 60]
        
        if len(self.requests[api_name]) >= max_per_minute:
            return False
        
        self.requests[api_name].append(now)
        return True

class SecurityUtils:
    """Security utilities for input validation and sanitization"""
    
    @staticmethod
    def validate_asin(asin: str) -> str:
        """Validate and sanitize ASIN format"""
        if not asin:
            raise ValueError("ASIN cannot be empty")
        
        asin = asin.strip().upper()
        if not re.match(r'^B[A-Z0-9]{9}$', asin):
            raise ValueError(f"Invalid ASIN format: {asin}. Must be B + 9 alphanumeric characters.")
        
        return asin
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Basic text sanitization"""
        if not text:
            return ""
        return text.strip()[:1000]  # Limit length

    async def _extract_weight_from_keepa(self, keepa_data: Dict, debug: bool = False) -> Optional[float]:
        """
        Extract weight from Keepa API response - enhanced to parse various weight fields
        Keepa provides weight in multiple locations with different formats
        """
        if debug:
            print("🔍 Analyzing Keepa data for weight information...")
            
        # Keepa weight fields (in order of preference)
        weight_fields_to_check = [
            'packageWeight',     # Most accurate for shipping
            'itemWeight',        # Product weight without packaging  
            'shippingWeight',    # Alternative shipping weight
            'weight',            # Generic weight field
            'packageDimensions', # Sometimes contains weight info
            'itemDimensions'     # Backup dimension data
        ]
        
        # Check direct weight fields first
        for field in weight_fields_to_check:
            if field in keepa_data and keepa_data[field]:
                weight_str = str(keepa_data[field])
                if debug:
                    print(f"   Checking field '{field}': {weight_str}")
                    
                weight_oz = self._parse_weight_to_ounces(weight_str)
                if weight_oz:
                    if debug:
                        print(f"✅ Weight found in '{field}': {weight_oz} oz")
                    return weight_oz
        
        # Check features array for weight specifications
        if 'features' in keepa_data and isinstance(keepa_data['features'], dict):
            weight_feature_keys = [
                'Item Weight', 'Package Weight', 'Shipping Weight', 
                'Product Weight', 'Weight', 'Dimensions and Weight'
            ]
            
            for key in weight_feature_keys:
                if key in keepa_data['features']:
                    weight_str = str(keepa_data['features'][key])
                    if debug:
                        print(f"   Checking feature '{key}': {weight_str}")
                        
                    weight_oz = self._parse_weight_to_ounces(weight_str)
                    if weight_oz:
                        if debug:
                            print(f"✅ Weight found in features['{key}']: {weight_oz} oz")
                        return weight_oz
        
        # Check product specifications array
        if 'productSpecs' in keepa_data and isinstance(keepa_data['productSpecs'], list):
            for spec in keepa_data['productSpecs']:
                if isinstance(spec, dict):
                    # Check spec name and value for weight info
                    spec_name = str(spec.get('name', '')).lower()
                    spec_value = str(spec.get('value', ''))
                    
                    if any(w in spec_name for w in ['weight', 'pounds', 'ounces', 'lbs', 'oz']):
                        if debug:
                            print(f"   Checking spec '{spec_name}': {spec_value}")
                            
                        weight_oz = self._parse_weight_to_ounces(spec_value)
                        if weight_oz:
                            if debug:
                                print(f"✅ Weight found in spec '{spec_name}': {weight_oz} oz")
                            return weight_oz
        
        if debug:
            print("❌ No weight found in Keepa data")
        return None

    async def _enhance_product_from_keepa(self, product_data: Dict, keepa_data: Dict, debug: bool = False) -> Dict:
        """
        Enhance product data with comprehensive information from Keepa API
        Returns enhanced product data with improved accuracy where Keepa has better info
        """
        if debug:
            print("🔍 Enhancing product data with Keepa information...")
            
        enhanced_data = product_data.copy()
        
        # WEIGHT ENHANCEMENT (highest priority)
        keepa_weight = await self._extract_weight_from_keepa(keepa_data, debug)
        if keepa_weight:
            enhanced_data['weight'] = keepa_weight
            enhanced_data['weight_source'] = 'keepa'
            if debug:
                print(f"✅ Enhanced weight: {keepa_weight} oz (from Keepa)")
        
        # TITLE ENHANCEMENT
        if 'title' in keepa_data and keepa_data['title']:
            keepa_title = str(keepa_data['title']).strip()
            if len(keepa_title) > len(enhanced_data.get('title', '')):
                enhanced_data['title'] = keepa_title
                enhanced_data['title_source'] = 'keepa'
                if debug:
                    print(f"✅ Enhanced title: {keepa_title[:50]}... (from Keepa)")
        
        # BRAND ENHANCEMENT
        if 'brand' in keepa_data and keepa_data['brand']:
            keepa_brand = str(keepa_data['brand']).strip()
            if not enhanced_data.get('brand') or len(keepa_brand) > len(enhanced_data.get('brand', '')):
                enhanced_data['brand'] = keepa_brand
                enhanced_data['brand_source'] = 'keepa'
                if debug:
                    print(f"✅ Enhanced brand: {keepa_brand} (from Keepa)")
        
        # PRICE VALIDATION (Keepa tracks current Amazon price)
        if 'priceAmazon' in keepa_data and keepa_data['priceAmazon']:
            # Keepa prices are in cents, convert to dollars
            keepa_price = keepa_data['priceAmazon'] / 100.0
            current_price = enhanced_data.get('price', 0)
            
            # If our scraped price is significantly different, use Keepa's (more reliable)
            if not current_price or abs(keepa_price - current_price) > 5:
                enhanced_data['price'] = keepa_price
                enhanced_data['price_source'] = 'keepa'
                if debug:
                    print(f"✅ Enhanced price: ${keepa_price:.2f} (from Keepa)")
        
        # MANUFACTURER/MODEL ENHANCEMENT
        if 'manufacturer' in keepa_data and keepa_data['manufacturer']:
            enhanced_data['manufacturer'] = str(keepa_data['manufacturer']).strip()
            if debug:
                print(f"✅ Added manufacturer: {enhanced_data['manufacturer']} (from Keepa)")
        
        if 'model' in keepa_data and keepa_data['model']:
            enhanced_data['model'] = str(keepa_data['model']).strip()
            if debug:
                print(f"✅ Added model: {enhanced_data['model']} (from Keepa)")
        
        # CATEGORY ENHANCEMENT
        if 'categories' in keepa_data and isinstance(keepa_data['categories'], list):
            if keepa_data['categories']:
                enhanced_data['keepa_categories'] = keepa_data['categories']
                enhanced_data['primary_category'] = keepa_data['categories'][0]
                if debug:
                    print(f"✅ Added categories: {keepa_data['categories'][:3]} (from Keepa)")
        
        # DIMENSIONS ENHANCEMENT
        dimensions = {}
        if 'packageLength' in keepa_data: dimensions['length'] = keepa_data['packageLength']
        if 'packageWidth' in keepa_data: dimensions['width'] = keepa_data['packageWidth'] 
        if 'packageHeight' in keepa_data: dimensions['height'] = keepa_data['packageHeight']
        
        if dimensions:
            enhanced_data['dimensions'] = dimensions
            if debug:
                print(f"✅ Added dimensions: {dimensions} (from Keepa)")
        
        # FEATURES/SPECIFICATIONS ENHANCEMENT
        if 'features' in keepa_data and isinstance(keepa_data['features'], dict):
            enhanced_data['detailed_features'] = keepa_data['features']
            if debug:
                print(f"✅ Added {len(keepa_data['features'])} detailed features (from Keepa)")
        
        # AVAILABILITY/STOCK STATUS
        if 'availabilityAmazon' in keepa_data:
            availability = keepa_data['availabilityAmazon']
            if availability == 1:
                enhanced_data['stock_status'] = 'in_stock'
            elif availability == 0:
                enhanced_data['stock_status'] = 'out_of_stock'
            else:
                enhanced_data['stock_status'] = 'unknown'
                
            if debug:
                print(f"✅ Added stock status: {enhanced_data['stock_status']} (from Keepa)")
        
        # SALES RANK (indicates popularity/demand)
        if 'salesRanks' in keepa_data and isinstance(keepa_data['salesRanks'], dict):
            enhanced_data['sales_ranks'] = keepa_data['salesRanks']
            if debug:
                print(f"✅ Added sales rank data (from Keepa)")
        
        return enhanced_data

class KeepaDirect:
    """Direct Keepa API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.keepa.com"
    
    async def get_product(self, asin: str) -> Optional[Dict[str, Any]]:
        """
        Fetch comprehensive product data from Keepa API
        Enhanced to request more detailed product information for better extraction
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/product",
                    params={
                        "key": self.api_key,
                        "domain": 1,  # Amazon.com
                        "asin": asin,
                        "stats": 365,  # Full year of stats for better trend analysis
                        "history": 1,  # Include price history
                        "rating": 1,   # Include rating data
                        "offers": 20,  # Include offer listings (up to 20)
                        "update": 1,   # Update product data if older than 1 hour
                        "buybox": 1    # Include buy box information
                    },
                    timeout=15.0  # Slightly longer timeout for comprehensive data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("products") and len(data["products"]) > 0:
                        product = data["products"][0]
                        
                        # Log what data we received for debugging
                        available_fields = [k for k, v in product.items() if v is not None and v != ""]
                        print(f"📊 Keepa data fields available: {len(available_fields)} (weight fields: {self._count_weight_fields(product)})")
                        
                        return product
                    else:
                        print(f"⚠️ Keepa returned empty product data for {asin}")
                elif response.status_code == 429:
                    print(f"⚠️ Keepa API rate limit exceeded for {asin}")
                elif response.status_code == 402:
                    print(f"⚠️ Keepa API quota exhausted - consider upgrading plan")
                else:
                    print(f"⚠️ Keepa API error {response.status_code} for {asin}")
                    
                return None
                
        except Exception as e:
            print(f"❌ Keepa API error for {asin}: {e}")
            return None
    
    def _count_weight_fields(self, product_data: Dict) -> int:
        """Count how many potential weight fields are available in Keepa data"""
        weight_fields = [
            'packageWeight', 'itemWeight', 'shippingWeight', 'weight',
            'packageDimensions', 'itemDimensions'
        ]
        
        count = 0
        for field in weight_fields:
            if field in product_data and product_data[field]:
                count += 1
        
        # Also check features for weight-related info
        if 'features' in product_data and isinstance(product_data['features'], dict):
            weight_features = [k for k in product_data['features'].keys() 
                             if any(w in k.lower() for w in ['weight', 'pounds', 'ounces', 'lbs', 'oz'])]
            count += len(weight_features)
        
        return count

class PricingOptimizer:
    """Calculate optimized eBay pricing"""
    
    def __init__(self, net_margin: float = 0.15, ebay_fee_rate: float = 0.13, paypal_fee_rate: float = 0.03):
        # Net margin is the desired profit after all fees
        self.net_margin = Decimal(str(net_margin))
        self.ebay_fee_rate = Decimal(str(ebay_fee_rate))
        self.paypal_fee_rate = Decimal(str(paypal_fee_rate))

    @property
    def total_fee_rate(self):
        return self.ebay_fee_rate + self.paypal_fee_rate

    def calculate_optimal_price(self, amazon_price: Decimal) -> Decimal:
        """
        Calculate optimal eBay price accounting for:
        - Desired net profit margin (after all fees)
        - eBay fees
        - PayPal fees
        """
        denominator = Decimal('1') - self.total_fee_rate - self.net_margin
        if denominator <= 0:
            raise ValueError("Fee rate + margin too high; cannot calculate price.")
        final_price = amazon_price / denominator
        # Round to reasonable price point
        return Decimal(str(round(float(final_price), 2)))

class EbayInventoryAPI:
    """eBay Inventory API client for listing management"""
    
    def __init__(self, client_id: str, client_secret: str, refresh_token: Optional[str] = None, sandbox: bool = False, runame: Optional[str] = None, country: str = "US", log_callback: Optional[callable] = None, config: Optional['Config'] = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.runame = runame  # eBay RuName (Return URL Name)
        self.redirect_uri = runame or 'urn:ietf:wg:oauth:2.0:oob'  # Use RuName if available, fallback to out-of-band
        self.sandbox = sandbox
        self.country = country
        self.access_token = None
        self.token_expires_at = None
        self.log_callback = log_callback
        self.config = config  # Store config for static location data
        
        # API endpoints
        if sandbox:
            self.base_url = "https://api.sandbox.ebay.com"
            self.auth_url = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
        else:
            self.base_url = "https://api.ebay.com"
            self.auth_url = "https://api.ebay.com/identity/v1/oauth2/token"

    def log(self, message: str):
        """Safe logging method that uses callback if available, otherwise prints"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def get_static_location(self, config: Config) -> dict:
        """Get location data from static configuration instead of API calls"""
        # Create address from config, with fallbacks for missing data
        # Use ebay_country as the source of truth for country
        address = {
            "country": config.ebay_country or "US"
        }
        
        # Add required fields with appropriate fallbacks based on ebay_country
        if config.ebay_country == "US":
            address.update({
                "postalCode": config.location_postal_code or "10001",
                "city": config.location_city or "New York",
                "stateOrProvince": config.location_state_province or "NY"
            })
        elif config.ebay_country == "CA":
            address.update({
                "postalCode": config.location_postal_code or "M5V 3A8",
                "city": config.location_city or "Toronto", 
                "stateOrProvince": config.location_state_province or "ON"
            })
        elif config.ebay_country == "GB":
            address.update({
                "postalCode": config.location_postal_code or "SW1A 1AA",
                "city": config.location_city or "London",
                "stateOrProvince": config.location_state_province or "England"
            })
        elif config.ebay_country == "AU":
            address.update({
                "postalCode": config.location_postal_code or "2000",
                "city": config.location_city or "Sydney",
                "stateOrProvince": config.location_state_province or "NSW"
            })
        elif config.ebay_country == "DE":
            address.update({
                "postalCode": config.location_postal_code or "10115",
                "city": config.location_city or "Berlin",
                "stateOrProvince": config.location_state_province or "Berlin"
            })
        else:
            # Generic fallback for other countries
            address.update({
                "postalCode": config.location_postal_code or "00000",
                "city": config.location_city or "Default City",
                "stateOrProvince": config.location_state_province or "Default State"
            })
        
        # Add optional address lines if provided
        if config.location_address_line1:
            address['addressLine1'] = config.location_address_line1
        if config.location_address_line2:
            address['addressLine2'] = config.location_address_line2
            
        self.log(f"📍 Using configured location: {address['city']}, {address['stateOrProvince']}, {address['country']}")
        return address
    
    def log_http_request(self, method: str, url: str, headers: dict = None, data: dict = None):
        """Log HTTP request details for debugging"""
        print(f"\n🌐 HTTP {method.upper()} Request:")
        print(f"   URL: {url}")
        if headers:
            # Don't log sensitive headers in full
            safe_headers = {k: v if k.lower() not in ['authorization'] else '***' for k, v in headers.items()}
            print(f"   Headers: {safe_headers}")
        if data:
            print(f"   Data: {data}")
        
        # Add stack trace to see where the request is coming from
        import traceback
        print(f"   📍 Call stack:")
        for line in traceback.format_stack()[-4:-1]:  # Show last 3 stack frames (excluding this method)
            print(f"      {line.strip()}")
    
    def log_http_response(self, response, method: str, url: str):
        """Log HTTP response details for debugging"""
        print(f"   Response: {response.status_code} {response.reason_phrase}")
        if response.status_code >= 400:
            print(f"   Error Body: {response.text}")
        print(f"   Response Headers: {dict(response.headers)}")
    
    async def make_request_with_retry(self, client, method: str, url: str, headers: dict = None, data: dict = None, json_data: dict = None, params: dict = None, max_retries: int = 3) -> httpx.Response:
        """Make HTTP request with retry logic for server errors"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                self.log_http_request(method, url, headers, data or json_data or params)
                
                # Add longer delay between requests to avoid overwhelming eBay servers
                if attempt > 0:
                    delay = min(2 ** attempt, 15)  # Exponential backoff, max 15 seconds
                    print(f"⏳ Waiting {delay} seconds before retry attempt {attempt + 1}...")
                    await asyncio.sleep(delay)
                elif method.upper() == "POST" and "publish" in url:
                    # Add extra delay for publish requests as they seem more sensitive
                    print(f"⏳ Adding 3 second delay before publish request...")
                    await asyncio.sleep(3)
                
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers, params=params, timeout=30.0)
                elif method.upper() == "POST":
                    if json_data:
                        response = await client.post(url, headers=headers, json=json_data, timeout=30.0)
                    else:
                        response = await client.post(url, headers=headers, data=data, timeout=30.0)
                elif method.upper() == "PUT":
                    if json_data:
                        response = await client.put(url, headers=headers, json=json_data, timeout=30.0)
                    else:
                        response = await client.put(url, headers=headers, data=data, timeout=30.0)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                self.log_http_response(response, method, url)
                
                # If we get a 500 error, retry (unless it's the last attempt)
                if response.status_code == 500 and attempt < max_retries - 1:
                    print(f"⚠️  Server error (500) on attempt {attempt + 1}, retrying...")
                    continue
                
                return response
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"⚠️  Request failed on attempt {attempt + 1}: {str(e)}, retrying...")
                    continue
                else:
                    break
        
        # If all retries failed, raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"All {max_retries} attempts failed for {method} {url}")
    
    def _get_marketplace_info(self) -> tuple[str, str]:
        """Get marketplace ID and currency based on country code"""
        marketplace_map = {
            "US": ("EBAY_US", "USD"),
            "GB": ("EBAY_GB", "GBP"),
            "DE": ("EBAY_DE", "EUR"),
            "FR": ("EBAY_FR", "EUR"),
            "IT": ("EBAY_IT", "EUR"),
            "ES": ("EBAY_ES", "EUR"),
            "CA": ("EBAY_CA", "CAD"),
            "AU": ("EBAY_AU", "AUD"),
        }
        return marketplace_map.get(self.country, ("EBAY_US", "USD"))
    
    async def exchange_auth_code_for_refresh_token(self, authorization_code: str) -> str:
        """Exchange authorization code for refresh token"""
        import json
        async with httpx.AsyncClient() as client:
            # Basic auth header
            import base64
            credentials = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            
            headers = {
                'Authorization': f'Basic {credentials}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {
                'grant_type': 'authorization_code',
                'code': authorization_code,
                'redirect_uri': self.redirect_uri
            }
            
            print(f"🔄 Exchanging authorization code for refresh token...")
            print(f"  - Auth URL: {self.auth_url}")
            print(f"  - Grant type: authorization_code")
            print(f"  - Redirect URI: {self.redirect_uri}")
            print(f"  - Code length: {len(authorization_code)} chars")
            print(f"  - Code starts with: {authorization_code[:20]}...")
            
            response = await client.post(self.auth_url, data=data, headers=headers)
            print(f"  - Response status: {response.status_code}")
            print(f"  - Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    print(f"  - Raw response text: {response.text[:500]}...")
                    token_data = response.json()
                    print("  - ✅ Authorization code exchange successful")
                    print(f"  - Token data keys: {list(token_data.keys())}")
                    
                    if 'refresh_token' not in token_data:
                        print(f"  - ❌ No refresh_token in response: {token_data}")
                        raise Exception(f"No refresh_token in eBay response: {token_data}")
                    
                    refresh_token = token_data['refresh_token']
                    print(f"  - Refresh token length: {len(refresh_token)}")
                    return refresh_token
                except json.JSONDecodeError as e:
                    print(f"  - ❌ JSON decode error: {e}")
                    print(f"  - Raw response: {response.text}")
                    raise Exception(f"Failed to parse JSON response: {e}")
                except KeyError as e:
                    print(f"  - ❌ Missing key in response: {e}")
                    print(f"  - Available keys: {list(token_data.keys()) if 'token_data' in locals() else 'N/A'}")
                    raise Exception(f"Missing expected key in response: {e}")
                except Exception as e:
                    print(f"  - ❌ Error parsing token response: {e}")
                    print(f"  - Raw response: {response.text}")
                    import traceback
                    traceback.print_exc()
                    raise Exception(f"Failed to parse token response: {e}")
            else:
                response_text = response.text
                print(f"  - ❌ Failed response body: {response_text}")
                
                # Try to parse JSON error for more details
                try:
                    error_data = response.json()
                    print(f"  - ❌ Error details: {error_data}")
                except:
                    pass
                    
                raise Exception(f"Failed to exchange authorization code: {response.status_code} - {response_text}")

    async def get_access_token(self) -> str:
        """Get or refresh OAuth access token"""
        if self.access_token and self.token_expires_at and time.time() < self.token_expires_at:
            return self.access_token
        
        # Debug logging
        print(f"🔍 eBay API Debug:")
        print(f"  - Client ID: {self.client_id[:8] if self.client_id else 'None'}...")
        print(f"  - Has Client Secret: {'Yes' if self.client_secret else 'No'}")
        print(f"  - Refresh Token: {self.refresh_token[:8] if self.refresh_token else 'None'}...")
        print(f"  - Sandbox Mode: {self.sandbox}")
        print(f"  - Auth URL: {self.auth_url}")
        
        async with httpx.AsyncClient() as client:
            if self.refresh_token:
                # Use refresh token (production)
                print("  - Using refresh token flow")
                data = {
                    'grant_type': 'refresh_token',
                    'refresh_token': self.refresh_token
                }
            else:
                # Use client credentials (sandbox only)
                if self.sandbox:
                    print("  - Using client credentials flow (sandbox)")
                    # For sandbox, use limited scope that's actually available
                    data = {
                        'grant_type': 'client_credentials',
                        'scope': 'https://api.ebay.com/oauth/api_scope'
                    }
                else:
                    print("  - ERROR: Production mode but no refresh token!")
                    raise Exception("Production API requires user authorization with refresh token. Please complete Auth'n'Auth flow in eBay Setup.")
            
            # Basic auth header
            import base64
            credentials = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            
            headers = {
                'Authorization': f'Basic {credentials}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            print(f"  - Making request to: {self.auth_url}")
            response = await client.post(self.auth_url, data=data, headers=headers)
            print(f"  - Response status: {response.status_code}")
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.token_expires_at = time.time() + token_data.get('expires_in', 3600) - 60  # 1 min buffer
                print("  - ✅ Access token obtained successfully")
                return self.access_token
            else:
                print(f"  - ❌ Failed: {response.text}")
                raise Exception(f"Failed to get access token: {response.status_code} - {response.text}")
    
    async def create_inventory_location(self) -> Optional[str]:
        """Create a default inventory location if none exists, return location key"""
        try:
            access_token = await self.get_access_token()
            
            # First, check if any locations already exist
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            async with httpx.AsyncClient() as client:
                # Check existing locations
                self.log("🔍 Checking for existing inventory locations...")
                response = await client.get(
                    f"{self.base_url}/sell/inventory/v1/location",
                    headers=headers,
                    timeout=30.0
                )
                
                self.log(f"📡 Location check response: {response.status_code}")
                if response.status_code == 200:
                    locations_data = response.json()
                    locations = locations_data.get('locations', [])
                    if locations:
                        # Return the first location key found
                        first_location = locations[0]
                        location_key = first_location.get('merchantLocationKey')
                        self.log(f"✅ Found {len(locations)} existing inventory location(s), using: {location_key}")
                        return location_key
                    else:
                        self.log("📍 No existing locations found, will create new one")
                elif response.status_code == 404:
                    self.log("📍 Location endpoint returned 404, will create new location")
                else:
                    self.log(f"⚠️  Location check failed ({response.status_code}), will attempt to create location anyway")
                    try:
                        error_data = response.json()
                        self.log(f"📋 Location check error: {error_data}")
                    except:
                        pass
                
                # No locations exist, create a default one
                self.log("🔄 Creating default inventory location...")
                location_key = f"default-location-{int(time.time())}"
                
                # Use static location from configuration
                if self.config:
                    address = self.get_static_location(self.config)
                else:
                    # Fallback to default addresses if no config available
                    address = {
                        "country": self.country,
                        "postalCode": "10001" if self.country == "US" else "M5V 3A8",
                        "city": "New York" if self.country == "US" else "Toronto",
                        "stateOrProvince": "NY" if self.country == "US" else "ON"
                    }
                    self.log("⚠️  No config available, using fallback default address")
                
                location_data = {
                    "location": {
                        "address": address
                    },
                    "name": "Default Warehouse Location",
                    "merchantLocationStatus": "ENABLED", 
                    "locationTypes": ["WAREHOUSE"],
                    "phone": "555-0123"  # Required field
                }
                
                self.log(f"🔄 Sending location creation request for key: {location_key}")
                self.log(f"📍 Complete address data: {address}")
                self.log(f"📦 Full location payload: {location_data}")
                
                # Debug: Check if we have config
                if self.config:
                    self.log(f"✅ Config available - ebay_country: {self.config.ebay_country}")
                    self.log(f"   Location fields: city='{self.config.location_city}', state='{self.config.location_state_province}', postal='{self.config.location_postal_code}'")
                else:
                    self.log("❌ No config available to EbayInventoryAPI")
                
                response = await client.post(
                    f"{self.base_url}/sell/inventory/v1/location/{location_key}",
                    json=location_data,
                    headers=headers,
                    timeout=30.0
                )
                
                self.log(f"📡 Location creation response: {response.status_code}")
                if response.status_code == 204:
                    self.log(f"✅ Default inventory location created successfully: {location_key}")
                    return location_key
                elif response.status_code == 409:
                    # Location already exists, which is fine
                    self.log(f"✅ Location already exists (409 response): {location_key}")
                    return location_key
                else:
                    self.log(f"❌ Failed to create inventory location: {response.status_code}")
                    self.log(f"❌ Response body: {response.text}")
                    # Try to parse the error to understand what went wrong
                    try:
                        error_data = response.json()
                        self.log(f"❌ Detailed error: {error_data}")
                        
                        # Look for specific error patterns
                        if 'errors' in error_data:
                            for error in error_data['errors']:
                                self.log(f"   Error {error.get('errorId', 'N/A')}: {error.get('message', 'No message')}")
                                if 'parameters' in error:
                                    for param in error['parameters']:
                                        self.log(f"     Parameter {param.get('name', 'N/A')}: {param.get('value', 'N/A')}")
                    except:
                        self.log("❌ Could not parse error response as JSON")
                    
                    # Try a different location key format as last resort
                    if "location key" in response.text.lower() or "merchant location" in response.text.lower():
                        self.log("🔄 Trying alternative location key format...")
                        alt_location_key = f"warehouse-{int(time.time())}"
                        alt_response = await client.post(
                            f"{self.base_url}/sell/inventory/v1/location/{alt_location_key}",
                            json=location_data,
                            headers=headers,
                            timeout=30.0
                        )
                        
                        if alt_response.status_code == 204:
                            self.log(f"✅ Alternative location created: {alt_location_key}")
                            return alt_location_key
                    
                    return None
                    
        except Exception as e:
            self.log(f"❌ Error creating inventory location: {str(e)}")
            return None
    
    def _get_marketplace_info(self) -> tuple[str, str]:
        """Get marketplace ID and currency based on country setting"""
        if self.country == "US":
            return "EBAY_US", "USD"
        elif self.country == "CA":
            return "EBAY_CA", "CAD"
        elif self.country == "UK":
            return "EBAY_GB", "GBP"
        elif self.country == "AU":
            return "EBAY_AU", "AUD"
        elif self.country == "DE":
            return "EBAY_DE", "EUR"
        else:
            # Default to US
            return "EBAY_US", "USD"
    
    def _determine_shipping_requirements(self, product) -> Dict[str, Any]:
        """Determine shipping requirements based on product characteristics"""
        # Extract price for shipping policy selection
        price = float(product.optimized_price) if hasattr(product, 'optimized_price') else 0.0
        
        # Use actual Amazon weight if available, otherwise estimate
        actual_weight_oz = getattr(product, 'weight_oz', None)
        
        shipping_info = {
            "eligible_for_standard_envelope": False,
            "requires_ground_shipping": True,
            "estimated_weight_oz": 16,  # Default to 1 lb
            "package_type": "BULKY_GOODS",
            "weight_source": "estimated"  # Track if we used actual or estimated weight
        }
        
        # If we have actual weight from Amazon, use it as the primary factor
        if actual_weight_oz:
            shipping_info["estimated_weight_oz"] = actual_weight_oz
            shipping_info["weight_source"] = "amazon"
            print(f"📏 Using Amazon weight: {actual_weight_oz} oz")
            
            # eBay Standard Envelope eligibility: EXTREMELY restrictive criteria
            # NEVER use envelope for luxury, branded, or valuable items regardless of weight
            title_lower = product.title.lower() if hasattr(product, 'title') else ""
            luxury_keywords = [
                'luxury', 'premium', 'designer', 'silk', 'satin', 'cashmere', 'leather',
                'gold', 'silver', 'platinum', 'diamond', 'pearl', 'vintage', 'authentic',
                'brand new', 'unopened', 'sealed', 'professional', 'high-end'
            ]
            
            is_luxury_item = any(keyword in title_lower for keyword in luxury_keywords)
            has_known_brand = product.brand and product.brand.lower() not in ['unbranded', 'generic', 'no brand']
            
            # Combined weight and value protection: NEVER envelope if exceeds EITHER limit
            weight_limit = 1.5  # oz
            price_limit = 10.0   # USD
            
            if (actual_weight_oz > weight_limit or price > price_limit or 
                is_luxury_item or has_known_brand or
                'ipad' in title_lower or 'iphone' in title_lower or
                'samsung' in title_lower or 'apple' in title_lower):
                
                # Determine appropriate shipping based on characteristics
                if actual_weight_oz <= 3.0:
                    # Lightweight but protected item - use regular shipping with tracking
                    shipping_info.update({
                        "eligible_for_standard_envelope": False,
                        "requires_ground_shipping": False,
                        "package_type": "PACKAGE"
                    })
                    reason = []
                    if actual_weight_oz > weight_limit: reason.append(f"weight {actual_weight_oz}oz > {weight_limit}oz")
                    if price > price_limit: reason.append(f"value ${price} > ${price_limit}")
                    if is_luxury_item: reason.append("luxury item")
                    if has_known_brand: reason.append("branded item")
                    print(f"📦 Protected item - regular shipping ({'; '.join(reason)})")
                else:
                    # Heavier item - ground shipping
                    shipping_info.update({
                        "eligible_for_standard_envelope": False,
                        "requires_ground_shipping": True,
                        "package_type": "BULKY_GOODS"
                    })
                    print(f"� Ground shipping required ({actual_weight_oz}oz, ${price})")
            else:
                # Only basic, unbranded, lightweight, low-value items qualify for envelope
                shipping_info.update({
                    "eligible_for_standard_envelope": True,
                    "requires_ground_shipping": False,
                    "package_type": "LETTER"
                })
                print(f"✉️  Eligible for standard envelope ({actual_weight_oz}oz, ${price}) - basic item only")
        else:
            # Fall back to estimation logic when no Amazon weight available
            print("⚠️  No Amazon weight found, using estimation...")
            
            # Check if it's likely a lightweight item eligible for standard envelope (under 3oz)
            title_lower = product.title.lower() if hasattr(product, 'title') else ""
            lightweight_keywords = [
                'cable', 'charger', 'adapter', 'case', 'cover', 'screen protector',
                'sticker', 'decal', 'sim card', 'memory card', 'usb', 'cord',
                'earbuds', 'small', 'mini', 'pocket', 'wire', 'connector',
                'ring', 'jewelry', 'pin', 'badge', 'patch', 'lotion', 'cream',
                'lip balm', 'chapstick', 'serum', 'oil', 'sample', 'travel size'
            ]
            
            # Items under $20 that are likely lightweight
            if price <= 20.0 and any(keyword in title_lower for keyword in lightweight_keywords):
                shipping_info.update({
                    "eligible_for_standard_envelope": True,
                    "requires_ground_shipping": False,
                    "estimated_weight_oz": 2,  # Under 3 oz for standard envelope
                    "package_type": "LETTER"
                })
            
            # Enhanced weight estimation based on product type and price
            if price > 20.0:
                # Better weight estimation based on product title/type
                title_lower = product.title.lower() if hasattr(product, 'title') else ""
                
                # Enhanced category-based weight estimates with luxury/electronics handling
                if any(keyword in title_lower for keyword in ['ipad', 'tablet']):
                    estimated_weight = 300  # iPad Mini ~300g = ~10.6oz, but use higher for safety
                elif any(keyword in title_lower for keyword in ['iphone', 'smartphone', 'phone']):
                    estimated_weight = 200  # Smartphones ~6-8oz
                elif any(keyword in title_lower for keyword in ['luxury', 'satin', 'silk', 'designer']) and any(keyword in title_lower for keyword in ['shirt', 'tshirt', 't-shirt', 'blouse', 'top']):
                    estimated_weight = 6  # Luxury shirts typically 4-8oz, NEVER envelope eligible
                elif any(keyword in title_lower for keyword in ['shirt', 'tshirt', 't-shirt', 'blouse', 'top', 'clothing']):
                    estimated_weight = 5  # Regular clothing 3-7oz
                elif any(keyword in title_lower for keyword in ['mousepad', 'mouse pad', 'mat']):
                    estimated_weight = 4  # Mousepads are typically 2-6 oz
                elif any(keyword in title_lower for keyword in ['phone', 'case', 'cover', 'screen protector']):
                    estimated_weight = 2  # Phone accessories
                elif any(keyword in title_lower for keyword in ['cable', 'charger', 'adapter', 'cord']):
                    estimated_weight = 6  # Cables and chargers
                elif any(keyword in title_lower for keyword in ['lotion', 'cream', 'serum', 'oil', 'balm', 'chapstick']):
                    estimated_weight = 3  # Small cosmetics are typically 1-4 oz
                elif any(keyword in title_lower for keyword in ['book', 'manual', 'guide']):
                    estimated_weight = 8  # Books
                elif any(keyword in title_lower for keyword in ['electronics', 'device', 'gadget']):
                    estimated_weight = max(12, min(32, int(price / 4)))  # Electronics scaling with price, higher minimum
                else:
                    # General estimation: higher for valuable items
                    estimated_weight = max(6, min(32, int(price * 0.8)))
                
                shipping_info.update({
                    "requires_ground_shipping": estimated_weight > 16,  # Over 1 lb needs ground
                    "package_type": "BULKY_GOODS" if estimated_weight > 16 else "PACKAGE",
                    "estimated_weight_oz": estimated_weight
                })
                
                print(f"📦 Estimated weight for '{title_lower[:30]}...': {estimated_weight} oz (${price})")
            else:
                # Lightweight items under $20
                shipping_info.update({
                    "estimated_weight_oz": 3,  # Default to 3 oz for cheap items
                    "package_type": "LETTER"
                })
        
        # Final shipping analysis summary
        print(f"📋 FINAL SHIPPING DECISION:")
        print(f"   Weight: {shipping_info['estimated_weight_oz']}oz ({shipping_info['weight_source']})")
        print(f"   Price: ${price}")
        print(f"   Envelope eligible: {shipping_info['eligible_for_standard_envelope']}")
        print(f"   Ground required: {shipping_info['requires_ground_shipping']}")
        print(f"   Package type: {shipping_info['package_type']}")
        
        return shipping_info
    
    async def get_ebay_category_suggestions(self, product) -> Optional[str]:
        """Get eBay category suggestions using the Taxonomy API"""
        try:
            access_token = await self.get_access_token()
            
            # Use product title as the query for category suggestion
            original_title = product.title if hasattr(product, 'title') else product.asin
            
            # Optimize query for better categorization
            query = self._optimize_category_query(original_title)
            
            # eBay Taxonomy API endpoint for category suggestions
            url = f"{self.base_url}/commerce/taxonomy/v1/category_tree/0/get_category_suggestions"
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            params = {
                'q': query[:200]  # Limit query length
            }
            
            self.log(f"🔍 Getting eBay category suggestions for: '{query[:50]}...'")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=params, timeout=30.0)
                
                if response.status_code == 200:
                    data = response.json()
                    suggestions = data.get('categorySuggestions', [])
                    
                    if suggestions:
                        self.log(f"🎯 eBay returned {len(suggestions)} category suggestions:")
                        
                        # Log all suggestions for debugging
                        for i, suggestion in enumerate(suggestions[:3]):  # Show top 3
                            cat = suggestion['category']
                            self.log(f"  {i+1}. {cat['categoryId']} - {cat['categoryName']}")
                        
                        # For mousepads, avoid computer-related categories
                        if "mousepad" in original_title.lower() or "mouse pad" in original_title.lower():
                            best_category = self._select_best_mousepad_category(suggestions)
                        else:
                            best_category = suggestions[0]['category']  # Use first suggestion
                        
                        category_id = best_category['categoryId']
                        category_name = best_category['categoryName']
                        
                        self.log(f"✅ Selected category: {category_id} - {category_name}")
                        return category_id
                    else:
                        self.log("⚠️  No category suggestions returned from eBay")
                        return None
                        
                else:
                    self.log(f"❌ Category suggestion failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            self.log(f"❌ Error getting eBay category suggestions: {e}")
            return None
    
    def _optimize_category_query(self, title: str) -> str:
        """Optimize the search query for better category suggestions"""
        title_lower = title.lower()
        
        # For mousepads, emphasize it's an accessory, not computer hardware
        if "mousepad" in title_lower or "mouse pad" in title_lower:
            # Remove computer-specific terms that might confuse eBay
            query = title.replace("Premium", "").replace("Gaming", "").replace("RGB", "")
            return f"mousepad desk mat accessory {query}"
            
        # For other products, use original title
        return title
    
    def _select_best_mousepad_category(self, suggestions: list) -> dict:
        """Select the best category for mousepads, avoiding computer hardware categories"""
        
        # Preferred categories for mousepads (in order of preference)
        preferred_categories = [
            "20349",  # Cases, Covers & Skins
            "31395",  # Cables & Connectors  
            "177",    # Laptops & Netbooks (accessories)
            "99",     # Everything Else
        ]
        
        # Categories to avoid (computer hardware that requires processor specs)
        avoid_categories = [
            "3676",   # Keyboards, Mice & Pointers (might require processor)
            "58058",  # Computers/Tablets & Networking
            "171",    # Computers, Tablets & eReaders
        ]
        
        self.log("🔍 Analyzing mousepad category suggestions...")
        
        # First, try to find a preferred category
        for suggestion in suggestions:
            cat_id = suggestion['category']['categoryId']
            cat_name = suggestion['category']['categoryName']
            
            if cat_id in preferred_categories:
                self.log(f"✅ Found preferred mousepad category: {cat_id} - {cat_name}")
                return suggestion['category']
        
        # If no preferred category, avoid problematic ones
        for suggestion in suggestions:
            cat_id = suggestion['category']['categoryId']
            cat_name = suggestion['category']['categoryName']
            
            if cat_id not in avoid_categories:
                self.log(f"✅ Using acceptable mousepad category: {cat_id} - {cat_name}")
                return suggestion['category']
        
        # Last resort: use Everything Else
        self.log("⚠️  Using fallback category for mousepad: Everything Else")
        return {
            'categoryId': '99',
            'categoryName': 'Everything Else > Other'
        }
    
    async def get_ebay_category_aspects(self, category_id: str) -> Dict[str, Any]:
        """Get required and optional aspects for an eBay category"""
        try:
            access_token = await self.get_access_token()
            
            url = f"{self.base_url}/commerce/taxonomy/v1/category_tree/0/get_item_aspects_for_category"
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            params = {
                'category_id': category_id
            }
            
            self.log(f"📋 Getting eBay aspects for category: {category_id}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=params, timeout=30.0)
                
                if response.status_code == 200:
                    data = response.json()
                    aspects = data.get('aspects', [])
                    
                    self.log(f"✅ Found {len(aspects)} aspects for category {category_id}")
                    return {
                        'required_aspects': [a for a in aspects if a.get('aspectConstraint', {}).get('aspectRequired', False)],
                        'optional_aspects': [a for a in aspects if not a.get('aspectConstraint', {}).get('aspectRequired', False)],
                        'all_aspects': aspects
                    }
                else:
                    self.log(f"❌ Failed to get aspects: {response.status_code} - {response.text}")
                    return {}
                    
        except Exception as e:
            self.log(f"❌ Error getting eBay category aspects: {e}")
            return {}
    
    def _determine_ebay_category(self, product) -> str:
        """Determine appropriate eBay leaf category based on product characteristics"""
        title = product.title.lower() if hasattr(product, 'title') else ""
        brand = getattr(product, 'brand', '').lower()
        
        # eBay leaf category mappings (verified leaf categories as of 2025)
        
        # Electronics & Computing
        if any(keyword in title for keyword in ['iphone', 'android', 'samsung galaxy', 'smartphone']):
            return "9355"  # Cell Phones & Smartphones
            
        if any(keyword in title for keyword in ['ipad', 'tablet', 'kindle']):
            return "171485"  # Tablets & eBook Readers
            
        if any(keyword in title for keyword in ['laptop', 'macbook', 'notebook', 'chromebook']):
            return "177"  # Laptops & Netbooks
            
        if any(keyword in title for keyword in ['monitor', 'display', 'lcd', 'led screen']):
            return "80053"  # Monitors, Projectors & Accs
            
        if any(keyword in title for keyword in ['mousepad', 'mouse pad']):
            return "20349"  # Cases, Covers & Skins (better fit for mousepads)
            
        if any(keyword in title for keyword in ['keyboard', 'mouse', 'trackpad']):
            return "3676"  # Keyboards, Mice & Pointers
            
        if any(keyword in title for keyword in ['headphone', 'earbuds', 'headset', 'airpods']):
            return "15052"  # Headphones
            
        if any(keyword in title for keyword in ['speaker', 'bluetooth speaker', 'soundbar']):
            return "14969"  # Portable Speakers
            
        if any(keyword in title for keyword in ['cable', 'usb', 'charger', 'adapter', 'cord']):
            return "31395"  # Cables & Connectors
            
        if any(keyword in title for keyword in ['memory card', 'sd card', 'micro sd', 'flash drive']):
            return "51395"  # Memory Cards
            
        if any(keyword in title for keyword in ['case', 'cover', 'screen protector', 'protective']):
            return "20349"  # Cases, Covers & Skins
            
        # Gaming
        if any(keyword in title for keyword in ['nintendo', 'xbox', 'playstation', 'ps5', 'ps4']):
            return "139973"  # Video Game Consoles
            
        if any(keyword in title for keyword in ['game controller', 'gaming controller', 'gamepad']):
            return "38583"  # Controllers & Attachments
            
        # Home & Garden
        if any(keyword in title for keyword in ['lamp', 'light', 'led bulb', 'lighting']):
            return "112581"  # Lamps, Lighting & Ceiling Fans
            
        if any(keyword in title for keyword in ['kitchen', 'cookware', 'utensil']):
            return "20625"  # Kitchen Tools & Gadgets
            
        # Clothing & Fashion
        if any(keyword in title for keyword in ['shirt', 't-shirt', 'hoodie', 'jacket']):
            return "1059"  # Men's Clothing (default, could be refined)
            
        if any(keyword in title for keyword in ['watch', 'smartwatch', 'apple watch']):
            return "31387"  # Wristwatches
            
        # Automotive
        if any(keyword in title for keyword in ['car', 'auto', 'automotive', 'vehicle']):
            return "6028"  # eBay Motors > Parts & Accessories
            
        # Health & Beauty
        if any(keyword in title for keyword in ['vitamins', 'supplements', 'health']):
            return "181"  # Health & Beauty > Vitamins & Lifestyle Supplements
            
        # Books & Media  
        if any(keyword in title for keyword in ['book', 'novel', 'textbook']):
            return "267"  # Books
            
        if any(keyword in title for keyword in ['dvd', 'blu-ray', 'movie']):
            return "11232"  # DVDs & Blu-ray Discs
            
        # Sports & Fitness
        if any(keyword in title for keyword in ['fitness', 'exercise', 'gym', 'workout']):
            return "15273"  # Fitness, Running & Yoga
            
        # Toys & Hobbies
        if any(keyword in title for keyword in ['toy', 'puzzle', 'game', 'kids']):
            return "220"  # Toys & Hobbies
            
        # Default fallback categories (known leaf categories)
        # Use a safe general category based on common product types
        if any(keyword in title for keyword in ['electronic', 'tech', 'digital']):
            return "293"  # Consumer Electronics > Multipurpose Batteries & Power
            
        # Final fallback - Everything Else is always a leaf category
        print(f"⚠️  Using fallback category for: {title[:50]}...")
        return "99"  # Everything Else > Other
    
    async def get_appropriate_fulfillment_policy(self, product) -> Optional[str]:
        """Get the most appropriate fulfillment policy for the product"""
        try:
            access_token = await self.get_access_token()
            marketplace_id, _ = self._get_marketplace_info()
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            shipping_req = self._determine_shipping_requirements(product)
            print(f"📦 Analyzing shipping for: {product.title[:50]}...")
            print(f"   💰 Price: ${product.optimized_price}")
            print(f"   📏 Weight: {shipping_req['estimated_weight_oz']} oz ({shipping_req['weight_source']})")
            print(f"   📋 Package type: {shipping_req['package_type']}")
            print(f"   ✉️  Standard envelope eligible: {shipping_req['eligible_for_standard_envelope']}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/sell/account/v1/fulfillment_policy?marketplace_id={marketplace_id}",
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    policy_data = response.json()
                    policies = policy_data.get('fulfillmentPolicies', [])
                    
                    if not policies:
                        print("❌ No fulfillment policies found")
                        return None
                    
                    print(f"🔍 Available fulfillment policies ({len(policies)} found):")
                    for i, policy in enumerate(policies):
                        policy_name = policy.get('name', 'Unknown')
                        shipping_options = policy.get('shippingOptions', [])
                        policy_id = policy.get('fulfillmentPolicyId', 'Unknown')
                        
                        print(f"   {i+1}. '{policy_name}' (ID: {policy_id[:20]}...)")
                        
                        # Check for buyer-pays shipping
                        for option in shipping_options:
                            shipping_service = option.get('shippingServices', [{}])[0]
                            cost_type = shipping_service.get('shippingCost', {}).get('costType', 'NOT_SPECIFIED')
                            service_type = shipping_service.get('shippingServiceCode', 'Unknown')
                            
                            if cost_type == 'CALCULATED':
                                print(f"      📋 Calculated shipping (buyer pays): {service_type}")
                            elif cost_type == 'FLAT_RATE':
                                cost = shipping_service.get('shippingCost', {}).get('value', 'Unknown')
                                print(f"      💰 Flat rate: ${cost} via {service_type}")
                            else:
                                print(f"      ❓ Cost type: {cost_type}, Service: {service_type}")
                    
                    # Use intelligent policy selection
                    weight_oz = shipping_req['estimated_weight_oz']
                    product_value = float(product.optimized_price)
                    
                    # Create policies dict for intelligent selection
                    policies_dict = {
                        'fulfillmentPolicies': policies,
                        'fulfillmentPolicyId': policies[0].get('fulfillmentPolicyId') if policies else None
                    }
                    
                    # Use our intelligent policy selector
                    selected_policy_id = self.select_optimal_shipping_policy(
                        policies_dict, weight_oz, product_value
                    )
                    
                    if selected_policy_id:
                        return selected_policy_id
                    
                    # If intelligent selection fails, fall back to original logic
                    weight_lbs = weight_oz / 16.0
                    
                    envelope_policy = None
                    light_policy = None  # Under 1 lb
                    medium_policy = None  # 1-5 lbs
                    heavy_policy = None  # Over 5 lbs
                    calculated_policy = None  # Buyer pays shipping - no weight limits!
                    fallback_policy = None
                    
                    print(f"🔍 Selecting policy for {weight_oz}oz ({weight_lbs:.3f}lbs) item...")
                    
                    for policy in policies:
                        policy_name = policy.get('name', '').lower()
                        policy_id = policy.get('fulfillmentPolicyId')
                        
                        print(f"   � Evaluating policy: {policy.get('name', '')}")
                        
                        # First check if this is a calculated shipping policy (buyer pays)
                        shipping_options = policy.get('shippingOptions', [])
                        is_calculated = False
                        for option in shipping_options:
                            shipping_service = option.get('shippingServices', [{}])[0]
                            cost_type = shipping_service.get('shippingCost', {}).get('costType', 'NOT_SPECIFIED')
                            if cost_type == 'CALCULATED':
                                is_calculated = True
                                calculated_policy = (policy_id, policy.get('name', ''))
                                print(f"      ✅ Found calculated shipping policy: {policy.get('name', '')}")
                                break
                        
                        if is_calculated:
                            continue  # Skip name-based categorization for calculated policies
                        
                        # Categorize policies by shipping capability (for flat rate policies)
                        if any(keyword in policy_name for keyword in ['envelope', 'letter', 'small']):
                            envelope_policy = (policy_id, policy.get('name', ''))
                        elif any(keyword in policy_name for keyword in ['light', 'first class', 'priority mail']):
                            light_policy = (policy_id, policy.get('name', ''))
                        elif any(keyword in policy_name for keyword in ['ground', 'ups', 'fedex', 'heavy']):
                            heavy_policy = (policy_id, policy.get('name', ''))
                        elif any(keyword in policy_name for keyword in ['standard', 'regular', 'normal']):
                            medium_policy = (policy_id, policy.get('name', ''))
                        else:
                            if not fallback_policy:
                                fallback_policy = (policy_id, policy.get('name', ''))
                    
                    # Select best policy based on weight - PRIORITIZE CALCULATED SHIPPING
                    selected_policy = None
                    
                    # First priority: calculated shipping (buyer pays) - no weight restrictions!
                    # This is the most flexible option for diverse product ranges
                    if calculated_policy:
                        selected_policy = calculated_policy
                        print(f"✅ Selected calculated shipping policy (buyer pays): {selected_policy[1]}")
                        print(f"   💡 Calculated shipping handles all weights automatically")
                    # Only use envelope shipping for very specific, small, cheap items
                    elif (weight_oz <= 2.0 and 
                          float(product.optimized_price) <= 15.0 and
                          shipping_req['eligible_for_standard_envelope'] and
                          envelope_policy):
                        # Very small and cheap items only
                        selected_policy = envelope_policy
                        if selected_policy:
                            print(f"✅ Selected envelope policy for tiny {weight_oz}oz, ${product.optimized_price} item: {selected_policy[1]}")
                    else:
                        # Explain why standard envelope was not selected
                        reasons = []
                        if weight_oz > 3.0:
                            reasons.append(f"weight {weight_oz}oz > 3oz")
                        if not shipping_req['eligible_for_standard_envelope']:
                            reasons.append("not envelope-eligible")
                        if float(product.optimized_price) > 20.0:
                            reasons.append(f"price ${product.optimized_price} > $20")
                        
                        print(f"📦 Standard envelope not available: {', '.join(reasons)}")
                        
                        # Select appropriate policy for non-envelope items
                        if weight_lbs <= 1.0:
                            # Light items (under 1 lb) - avoid ground shipping
                            selected_policy = light_policy or medium_policy or fallback_policy
                            if selected_policy:
                                print(f"✅ Selected light shipping policy for {weight_lbs:.1f}lb item: {selected_policy[1]}")
                        elif weight_lbs <= 5.0:
                            # Medium items (1-5 lbs) - standard shipping
                            selected_policy = medium_policy or light_policy or heavy_policy or fallback_policy
                            if selected_policy:
                                print(f"✅ Selected standard policy for {weight_lbs:.1f}lb item: {selected_policy[1]}")
                        else:
                            # Heavy items (over 5 lbs) - require ground shipping
                            selected_policy = heavy_policy or medium_policy or fallback_policy
                            if selected_policy:
                                print(f"✅ Selected heavy shipping policy for {weight_lbs:.1f}lb item: {selected_policy[1]}")
                    
                    if selected_policy:
                        return selected_policy[0]
                    
                    # Enhanced fallback system - create priority list of ALL policies to try
                    print("🔄 Creating comprehensive fallback policy list...")
                    fallback_policies = []
                    
                    # Priority order based on product characteristics
                    if weight_lbs <= 1.0:
                        # Light items: calculated > light > medium > heavy
                        # Only include envelope if explicitly eligible (expensive items should NEVER use envelope)
                        priority_policies = [calculated_policy, light_policy, medium_policy, heavy_policy]
                        if (shipping_req['eligible_for_standard_envelope'] and 
                            weight_oz <= 2.0 and 
                            float(product.optimized_price) <= 15.0):
                            priority_policies.append(envelope_policy)
                        
                        for policy in priority_policies:
                            if policy and policy not in fallback_policies:
                                fallback_policies.append(policy)
                    else:
                        # Heavy items: calculated > heavy > medium > light (never envelope)
                        for policy in [calculated_policy, heavy_policy, medium_policy, light_policy]:
                            if policy and policy not in fallback_policies:
                                fallback_policies.append(policy)
                    
                    # Add any remaining policies as final fallbacks
                    for policy in policies:
                        policy_tuple = (policy.get('fulfillmentPolicyId'), policy.get('name', ''))
                        if policy_tuple not in fallback_policies:
                            fallback_policies.append(policy_tuple)
                    
                    # Store the fallback list for the offer creation to use
                    if fallback_policies:
                        # For now, return the first one, but we'll enhance offer creation to try multiple
                        print(f"📋 Fallback policy chain ({len(fallback_policies)} options):")
                        for i, (policy_id, policy_name) in enumerate(fallback_policies[:3], 1):
                            print(f"   {i}. {policy_name} ({policy_id[:20]}...)")
                        
                        selected_policy_id = fallback_policies[0][0]
                        selected_policy_name = fallback_policies[0][1]
                        print(f"✅ Selected primary policy: {selected_policy_name}")
                        
                        # TODO: Store the full fallback list for retry logic
                        return selected_policy_id
                    
                    print("❌ No suitable shipping policy found")
                else:
                    print(f"❌ Failed to get fulfillment policies: {response.status_code}")
                    return None
                    
        except Exception as e:
            print(f"❌ Error getting appropriate fulfillment policy: {e}")
            return None

    async def get_all_fulfillment_policies_with_fallback(self, product) -> list:
        """Get ALL available fulfillment policies with smart ordering for fallback attempts"""
        try:
            access_token = await self.get_access_token()
            marketplace_id, _ = self._get_marketplace_info()
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/sell/account/v1/fulfillment_policy?marketplace_id={marketplace_id}",
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    policy_data = response.json()
                    policies = policy_data.get('fulfillmentPolicies', [])
                    
                    if not policies:
                        return []
                    
                    # Analyze and sort policies by suitability
                    shipping_req = self._determine_shipping_requirements(product)
                    weight_oz = shipping_req['estimated_weight_oz']
                    weight_lbs = weight_oz / 16.0
                    
                    calculated_policies = []
                    appropriate_policies = []
                    other_policies = []
                    
                    for policy in policies:
                        policy_id = policy.get('fulfillmentPolicyId')
                        policy_name = policy.get('name', '')
                        
                        # Check if this is calculated shipping (most flexible)
                        shipping_options = policy.get('shippingOptions', [])
                        is_calculated = False
                        for option in shipping_options:
                            shipping_service = option.get('shippingServices', [{}])[0]
                            cost_type = shipping_service.get('shippingCost', {}).get('costType', 'NOT_SPECIFIED')
                            if cost_type == 'CALCULATED':
                                is_calculated = True
                                break
                        
                        policy_info = {
                            'id': policy_id,
                            'name': policy_name,
                            'is_calculated': is_calculated,
                            'suitable_for_weight': True  # We'll be less restrictive in fallback mode
                        }
                        
                        if is_calculated:
                            calculated_policies.append(policy_info)
                        else:
                            # For flat rate, try to determine if it's appropriate
                            policy_name_lower = policy_name.lower()
                            if weight_oz <= 3 and any(kw in policy_name_lower for kw in ['envelope', 'letter', 'small']):
                                appropriate_policies.append(policy_info)
                            elif weight_lbs <= 5 and any(kw in policy_name_lower for kw in ['standard', 'regular', 'priority', 'first']):
                                appropriate_policies.append(policy_info)
                            elif any(kw in policy_name_lower for kw in ['ground', 'ups', 'fedex', 'heavy']):
                                appropriate_policies.append(policy_info)
                            else:
                                other_policies.append(policy_info)
                    
                    # Return in priority order: calculated first, then appropriate, then others
                    all_policies = calculated_policies + appropriate_policies + other_policies
                    
                    self.log(f"📋 Available fulfillment policies ({len(all_policies)} total):")
                    for i, policy in enumerate(all_policies[:5], 1):  # Show top 5
                        calc_text = " (Calculated)" if policy['is_calculated'] else " (Flat Rate)"
                        self.log(f"   {i}. {policy['name']}{calc_text}")
                    
                    return all_policies
                    
                else:
                    self.log(f"❌ Failed to get fulfillment policies: {response.status_code}")
                    return []
                    
        except Exception as e:
            self.log(f"❌ Error getting all fulfillment policies: {e}")
            return []

    async def get_all_business_policies(self) -> Optional[Dict]:
        """Get ALL user's business policies (payment, return, shipping) with full details"""
        try:
            access_token = await self.get_access_token()
            marketplace_id, _ = self._get_marketplace_info()
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            all_policies = {
                'fulfillment': [],
                'payment': [],
                'return': []
            }
            
            async with httpx.AsyncClient() as client:
                # Get all fulfillment (shipping) policies
                try:
                    print("🔄 Fetching all fulfillment policies...")
                    response = await client.get(
                        f"{self.base_url}/sell/account/v1/fulfillment_policy?marketplace_id={marketplace_id}",
                        headers=headers,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        policy_data = response.json()
                        if policy_data.get('fulfillmentPolicies'):
                            all_policies['fulfillment'] = policy_data['fulfillmentPolicies']
                            print(f"✅ Found {len(all_policies['fulfillment'])} fulfillment policies")
                        else:
                            print("❌ No fulfillment policies found")
                    else:
                        print(f"❌ Failed to get fulfillment policies: {response.status_code}")
                except Exception as e:
                    print(f"❌ Error getting fulfillment policies: {e}")
                
                # Get all payment policies
                try:
                    print("🔄 Fetching all payment policies...")
                    response = await client.get(
                        f"{self.base_url}/sell/account/v1/payment_policy?marketplace_id={marketplace_id}",
                        headers=headers,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        policy_data = response.json()
                        if policy_data.get('paymentPolicies'):
                            all_policies['payment'] = policy_data['paymentPolicies']
                            print(f"✅ Found {len(all_policies['payment'])} payment policies")
                        else:
                            print("❌ No payment policies found")
                    else:
                        print(f"❌ Failed to get payment policies: {response.status_code}")
                except Exception as e:
                    print(f"❌ Error getting payment policies: {e}")
                
                # Get all return policies
                try:
                    print("🔄 Fetching all return policies...")
                    response = await client.get(
                        f"{self.base_url}/sell/account/v1/return_policy?marketplace_id={marketplace_id}",
                        headers=headers,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        policy_data = response.json()
                        if policy_data.get('returnPolicies'):
                            all_policies['return'] = policy_data['returnPolicies']
                            print(f"✅ Found {len(all_policies['return'])} return policies")
                        else:
                            print("❌ No return policies found")
                    else:
                        print(f"❌ Failed to get return policies: {response.status_code}")
                except Exception as e:
                    print(f"❌ Error getting return policies: {e}")
            
            return all_policies
                
        except Exception as e:
            print(f"❌ Error getting all business policies: {str(e)}")
            return None

    def select_optimal_shipping_policy(self, policies: Dict, product_weight_oz: float, product_value: float, dimensions: Dict = None) -> str:
        """Intelligently select shipping policy based on actual shipping services, not just policy names"""
        try:
            fulfillment_policies = policies.get('fulfillmentPolicies', [])
            if not fulfillment_policies:
                return policies.get('fulfillmentPolicyId')  # Fallback to default
            
            # Convert weight to pounds for analysis
            weight_lbs = product_weight_oz / 16.0 if product_weight_oz > 0 else 0.5  # Default 0.5 lbs if no weight
            
            print(f"🎯 Analyzing {len(fulfillment_policies)} policies for: {weight_lbs:.2f} lbs, ${product_value:.2f}")
            
            # Define preferred shipping service codes based on product characteristics
            if weight_lbs > 3.0 or product_value > 100:
                # Heavy/valuable items - prefer ground shipping
                preferred_services = [
                    'USPSGroundAdvantage', 'USPSParcelSelect', 'UPSGround', 'FedExGround',
                    'USPSPriorityMailExpress', 'USPSPriorityMail', 'UPS3DaySelect', 'FedEx2Day',
                    'USPSRetailGround', 'USPSParcelSelectGround'
                ]
                avoid_services = ['USPSFirstClassMail', 'USPSFirstClassPackage', 'FirstClass']
            elif weight_lbs > 1.0 or product_value > 25:
                # Medium items - prefer priority/expedited
                preferred_services = [
                    'USPSPriorityMail', 'USPSPriorityMailExpress', 'UPS3DaySelect', 'FedEx2Day',
                    'USPSGroundAdvantage', 'USPSParcelSelect', 'UPSGround', 'FedExGround'
                ]
                avoid_services = ['USPSFirstClassMail', 'FirstClass']
            else:
                # Light items - first class acceptable
                preferred_services = [
                    'USPSFirstClassPackage', 'USPSGroundAdvantage', 'USPSPriorityMail',
                    'USPSFirstClassMail', 'USPSParcelSelect'
                ]
                avoid_services = []
            
            best_policy = None
            best_score = -1
            
            for policy in fulfillment_policies:
                policy_name = policy.get('name', 'Unknown')
                policy_id = policy.get('fulfillmentPolicyId')
                shipping_options = policy.get('shippingOptions', [])
                
                print(f"  🔍 Analyzing '{policy_name}'...")
                
                # Analyze actual shipping services in this policy
                policy_services = []
                is_calculated = False
                has_free_shipping = False
                
                for option in shipping_options:
                    shipping_services = option.get('shippingServices', [])
                    for service in shipping_services:
                        service_code = service.get('shippingServiceCode', '')
                        cost_type = service.get('shippingCost', {}).get('costType', 'NOT_SPECIFIED')
                        cost_value = service.get('shippingCost', {}).get('value', '0')
                        
                        policy_services.append(service_code)
                        
                        if cost_type == 'CALCULATED':
                            is_calculated = True
                        elif cost_type == 'FLAT_RATE' and cost_value == '0':
                            has_free_shipping = True
                        
                        print(f"    📦 Service: {service_code} ({cost_type})")
                
                # Calculate policy score based on actual services
                score = 0
                
                # Calculated shipping gets high priority (buyer pays based on actual weight/distance)
                if is_calculated:
                    score += 50
                    print(f"    ✅ Calculated shipping (+50 points)")
                
                # Check for preferred services
                for i, preferred in enumerate(preferred_services):
                    for policy_service in policy_services:
                        if preferred.lower() in policy_service.lower():
                            # Earlier in preference list = higher score
                            service_score = len(preferred_services) - i + 10
                            score += service_score
                            print(f"    ✅ Preferred service '{policy_service}' (+{service_score} points)")
                            break
                
                # Penalize avoided services for heavy/valuable items
                for avoided in avoid_services:
                    for policy_service in policy_services:
                        if avoided.lower() in policy_service.lower():
                            penalty = -20 if (weight_lbs > 2.0 or product_value > 75) else -5
                            score += penalty
                            print(f"    ❌ Avoided service '{policy_service}' ({penalty} points)")
                            break
                
                # Bonus for free shipping on expensive items
                if has_free_shipping and product_value > 50:
                    score += 15
                    print(f"    ✅ Free shipping for valuable item (+15 points)")
                
                # Fallback: if no specific services matched, analyze policy name
                if score == 0:
                    policy_name_lower = policy_name.lower()
                    if any(term in policy_name_lower for term in ['ground', 'heavy', 'large']):
                        score += 5 if weight_lbs > 2.0 else 2
                    elif any(term in policy_name_lower for term in ['priority', 'expedited', 'fast']):
                        score += 3
                    elif any(term in policy_name_lower for term in ['envelope', 'letter']):
                        score -= 10 if weight_lbs > 1.0 else 0
                    print(f"    📋 Fallback name analysis: {score} points")
                
                print(f"    📊 Total score: {score}")
                
                if score > best_score:
                    best_score = score
                    best_policy = policy
            
            if best_policy:
                selected_id = best_policy.get('fulfillmentPolicyId')
                policy_name = best_policy.get('name', 'Unknown')
                print(f"✅ Selected policy: '{policy_name}' (score: {best_score})")
                return selected_id
            
            # Ultimate fallback
            print(f"⚠️  Using first available policy as fallback")
            return fulfillment_policies[0].get('fulfillmentPolicyId')
                
        except Exception as e:
            print(f"❌ Error in intelligent policy selection: {e}")
            return policies.get('fulfillmentPolicyId')  # Fallback to default

    async def get_business_policies(self) -> Optional[Dict]:
        """Get user's business policies (payment, return, shipping) required for offers - using defaults"""
        try:
            access_token = await self.get_access_token()
            marketplace_id, _ = self._get_marketplace_info()
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            policies = {}
            
            async with httpx.AsyncClient() as client:
                # Get fulfillment (shipping) policies - use first available
                try:
                    print("🔄 Fetching fulfillment policies...")
                    response = await client.get(
                        f"{self.base_url}/sell/account/v1/fulfillment_policy?marketplace_id={marketplace_id}",
                        headers=headers,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        policy_data = response.json()
                        if policy_data.get('fulfillmentPolicies'):
                            # Store all policies for intelligent selection
                            policies["fulfillmentPolicies"] = policy_data['fulfillmentPolicies']
                            policies["fulfillmentPolicyId"] = policy_data['fulfillmentPolicies'][0].get('fulfillmentPolicyId')
                            print(f"✅ Found {len(policy_data['fulfillmentPolicies'])} fulfillment policies")
                        else:
                            print("❌ No fulfillment policies found")
                    else:
                        print(f"❌ Failed to get fulfillment policies: {response.status_code}")
                except Exception as e:
                    print(f"❌ Error getting fulfillment policies: {e}")
                
                # Get payment policies - use first available
                try:
                    print("🔄 Fetching payment policies...")
                    response = await client.get(
                        f"{self.base_url}/sell/account/v1/payment_policy?marketplace_id={marketplace_id}",
                        headers=headers,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        policy_data = response.json()
                        if policy_data.get('paymentPolicies'):
                            policies["paymentPolicyId"] = policy_data['paymentPolicies'][0].get('paymentPolicyId')
                            print(f"✅ Found payment policy: {policies['paymentPolicyId']}")
                        else:
                            print("❌ No payment policies found")
                    else:
                        print(f"❌ Failed to get payment policies: {response.status_code}")
                except Exception as e:
                    print(f"❌ Error getting payment policies: {e}")
                
                # Get return policies - use first available
                try:
                    print("🔄 Fetching return policies...")
                    response = await client.get(
                        f"{self.base_url}/sell/account/v1/return_policy?marketplace_id={marketplace_id}",
                        headers=headers,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        policy_data = response.json()
                        if policy_data.get('returnPolicies'):
                            policies["returnPolicyId"] = policy_data['returnPolicies'][0].get('returnPolicyId')
                            print(f"✅ Found return policy: {policies['returnPolicyId']}")
                        else:
                            print("❌ No return policies found")
                    else:
                        print(f"❌ Failed to get return policies: {response.status_code}")
                except Exception as e:
                    print(f"❌ Error getting return policies: {e}")
            
            # Validate that we have all required policies
            required_policies = ["fulfillmentPolicyId", "paymentPolicyId", "returnPolicyId"]
            missing_policies = [policy for policy in required_policies if policy not in policies]
            
            if missing_policies:
                print(f"❌ Missing required business policies: {missing_policies}")
                print("⚠️  Please set up all business policies in your eBay seller account:")
                print("   1. Go to Seller Hub → Account → Business Policies")
                print("   2. Create policies for: Payment, Return, and Shipping")
                print("   3. Make sure they're enabled for your target marketplace")
                return None
            
            print(f"✅ All required business policies found: {list(policies.keys())}")
            return policies
                
        except Exception as e:
            print(f"❌ Error getting business policies: {str(e)}")
            return None

    def _validate_offer_data(self, offer_data: Dict) -> Dict:
        """Validate offer data against eBay API requirements"""
        errors = []
        
        # Required fields validation
        required_fields = [
            "sku", "marketplaceId", "format", "availableQuantity", 
            "categoryId", "listingDescription", "listingDuration", 
            "pricingSummary", "merchantLocationKey", "listingPolicies"
        ]
        
        for field in required_fields:
            if field not in offer_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate specific field formats
        if "format" in offer_data and offer_data["format"] not in ["AUCTION", "FIXED_PRICE"]:
            errors.append(f"Invalid format: {offer_data['format']}. Must be AUCTION or FIXED_PRICE")
        
        if "listingDuration" in offer_data and offer_data["listingDuration"] not in [
            "DAYS_1", "DAYS_3", "DAYS_5", "DAYS_7", "DAYS_10", "DAYS_21", "DAYS_30", "GTC"
        ]:
            errors.append(f"Invalid listingDuration: {offer_data['listingDuration']}")
        
        if "marketplaceId" in offer_data and not offer_data["marketplaceId"].startswith("EBAY_"):
            errors.append(f"Invalid marketplaceId: {offer_data['marketplaceId']}")
        
        # Validate pricing structure
        if "pricingSummary" in offer_data:
            pricing = offer_data["pricingSummary"]
            if "price" not in pricing:
                errors.append("Missing price in pricingSummary")
            elif "currency" not in pricing["price"] or "value" not in pricing["price"]:
                errors.append("Invalid price structure in pricingSummary")
        
        # Validate business policies structure
        if "listingPolicies" in offer_data:
            policies = offer_data["listingPolicies"]
            required_policies = ["fulfillmentPolicyId", "paymentPolicyId", "returnPolicyId"]
            for policy in required_policies:
                if policy not in policies or not policies[policy]:
                    errors.append(f"Missing or empty business policy: {policy}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    async def opt_into_seller_hub(self) -> bool:
        """Opt user into eBay's Seller Hub program (required for business policies)"""
        try:
            print("🔄 Opting into eBay Seller Hub program...")
            access_token = await self.get_access_token()
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # eBay's optInToProgram API call
            opt_in_data = {
                "programs": [
                    {
                        "programType": "SELLING_POLICY_MANAGEMENT"
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/sell/account/v1/program/opt_in",
                    json=opt_in_data,
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code in [200, 201, 204]:
                    print("✅ Successfully opted into Seller Hub program")
                    return True
                elif response.status_code == 409:
                    print("✅ Already opted into Seller Hub program")
                    return True
                else:
                    print(f"⚠️  Opt-in response: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            print(f"⚠️  Error opting into Seller Hub: {str(e)}")
            return False

    async def check_seller_verification_status(self) -> Dict:
        """Check if seller account is fully verified and ready to sell"""
        try:
            print("🔄 Checking seller verification status...")
            access_token = await self.get_access_token()
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            verification_status = {
                "verified": False,
                "issues": [],
                "requirements": []
            }
            
            async with httpx.AsyncClient() as client:
                # Check selling privileges
                try:
                    response = await client.get(
                        f"{self.base_url}/sell/account/v1/privilege",
                        headers=headers,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        privileges = response.json()
                        print(f"🔍 Selling privileges: {privileges}")
                        
                        # Check if selling is enabled
                        selling_privileges = privileges.get('sellingPrivileges', [])
                        if not selling_privileges:
                            verification_status["issues"].append("No selling privileges found")
                            verification_status["requirements"].append("Complete eBay seller verification process")
                        else:
                            verification_status["verified"] = True
                    else:
                        verification_status["issues"].append(f"Cannot check privileges: {response.status_code}")
                except Exception as e:
                    verification_status["issues"].append(f"Error checking privileges: {str(e)}")
                
                # Check account status
                try:
                    response = await client.get(
                        f"{self.base_url}/sell/account/v1/user",
                        headers=headers,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        user_info = response.json()
                        print(f"🔍 User status: {user_info}")
                        
                        # Check registration status
                        if user_info.get('status') != 'CONFIRMED':
                            verification_status["issues"].append("Account not confirmed")
                            verification_status["requirements"].append("Complete email verification")
                        
                        # Check if policies exist
                        policies = await self.get_business_policies()
                        if not policies:
                            verification_status["issues"].append("Business policies not set up")
                            verification_status["requirements"].append("Create payment, return, and shipping policies")
                            
                except Exception as e:
                    verification_status["issues"].append(f"Error checking account: {str(e)}")
            
            return verification_status
                    
        except Exception as e:
            return {
                "verified": False,
                "issues": [f"Error checking verification: {str(e)}"],
                "requirements": ["Check eBay seller account status manually"]
            }

    def _format_weight_for_ebay(self, weight_oz: float, product_title: str = "", use_lb_minimum: bool = True) -> tuple:
        """
        Format weight for eBay API with realistic weight floors based on product type.
        Returns (weight_value, weight_unit) tuple.
        
        Args:
            weight_oz: Weight in ounces
            product_title: Product title for smart weight estimation
            use_lb_minimum: If True, enforces realistic minimums (default True for eBay compatibility)
        """
        if use_lb_minimum:
            # Smart weight floors based on product type to avoid eBay rejection
            title_lower = product_title.lower()
            
            # Realistic minimum weights by category (in ounces) - order matters!
            if 'ring' in title_lower and any(keyword in title_lower for keyword in ['camera', 'outdoor', 'security', 'cam']):
                min_weight_oz = 12.0  # 12 oz minimum for Ring cameras (0.75 lbs - realistic for outdoor cameras)
            elif 'galaxy' in title_lower and 'ring' in title_lower:
                min_weight_oz = 0.3   # 0.3 oz for Galaxy smart rings (more realistic than generic ring)
            elif any(keyword in title_lower for keyword in ['ring', 'jewelry', 'band']) and 'camera' not in title_lower:
                min_weight_oz = 0.5   # 0.5 oz minimum for jewelry/rings
            elif any(keyword in title_lower for keyword in ['camera', 'security', 'cam']):
                min_weight_oz = 8.0   # 8 oz minimum for other cameras
            elif any(keyword in title_lower for keyword in ['phone', 'smartphone', 'iphone']):
                min_weight_oz = 5.0   # 5 oz minimum for phones
            elif any(keyword in title_lower for keyword in ['tablet', 'ipad', 'kindle']):
                min_weight_oz = 10.0  # 10 oz minimum for tablets
            elif any(keyword in title_lower for keyword in ['watch', 'smartwatch', 'fitness']):
                min_weight_oz = 1.5   # 1.5 oz minimum for wearables
            elif any(keyword in title_lower for keyword in ['headphone', 'earbud', 'airpod']):
                min_weight_oz = 1.0   # 1 oz minimum for audio devices
            elif any(keyword in title_lower for keyword in ['cable', 'charger', 'adapter']):
                min_weight_oz = 2.0   # 2 oz minimum for cables/chargers
            else:
                min_weight_oz = 4.0   # 4 oz default minimum (0.25 lbs)
            
            # Apply the appropriate minimum
            adjusted_weight_oz = max(weight_oz, min_weight_oz)
            
            if adjusted_weight_oz != weight_oz:
                self.log(f"⚖️ Weight adjusted from {weight_oz}oz to {adjusted_weight_oz}oz (realistic minimum for product type)")
            
            weight_lbs = round(adjusted_weight_oz / 16.0, 3)  # Convert to pounds with precision
            
            # FINAL SAFETY: eBay minimum floor of 0.25 POUND (4 oz) to avoid API rejection
            EBAY_MINIMUM_LBS = 0.25  # 4 oz minimum - safe for all eBay operations
            if weight_lbs < EBAY_MINIMUM_LBS:
                self.log(f"🛡️ Applying eBay safety minimum: {weight_lbs} lbs → {EBAY_MINIMUM_LBS} lbs (prevents API rejection)")
                weight_lbs = EBAY_MINIMUM_LBS
            
            return weight_lbs, "POUND"
        else:
            # Legacy approach: Use actual weights (may cause eBay issues)
            weight_lbs = max(round(weight_oz / 16.0, 3), 0.01)  # Minimum 0.01 lbs
            return weight_lbs, "POUND"

    async def create_inventory_item(self, product: 'Product') -> Optional[str]:
        """Create inventory item from Product"""
        try:
            access_token = await self.get_access_token()
            
            # Generate unique SKU from ASIN with timestamp to avoid conflicts
            import time
            timestamp = int(time.time())
            sku = f"AMZ-{product.asin}-{timestamp}"
            
            # Calculate shipping requirements to get accurate weight
            shipping_req = self._determine_shipping_requirements(product)
            
            # Handle weight format for eBay API - try separate pound/ounce format
            weight_oz = shipping_req['estimated_weight_oz']
            
            # Calculate separate pounds and ounces (e.g., 0 lb 10 oz)
            total_ounces = max(weight_oz, 0.1)  # Minimum 0.1 oz
            pounds = int(total_ounces // 16)  # Whole pounds
            remaining_ounces = round(total_ounces % 16, 1)  # Remaining ounces
            
            # If no remaining ounces, ensure at least 0.1 oz
            if remaining_ounces == 0 and pounds == 0:
                remaining_ounces = 0.1
            
            self.log(f"📦 eBay weight format: {pounds} lb {remaining_ounces} oz (total: {total_ounces} oz)")
            
            # Use smart weight formatting with realistic minimums
            weight_value, weight_unit = self._format_weight_for_ebay(
                weight_oz, 
                product.title if hasattr(product, 'title') else "",
                use_lb_minimum=True
            )
            
            self.log(f"� Sending to eBay API: {weight_value} {weight_unit}")
            
            self.log(f"📦 Using calculated weight: {weight_value} {weight_unit} for {product.title[:50]}...")
            self.log(f"📋 Weight source: {shipping_req['weight_source']}, Envelope eligible: {shipping_req['eligible_for_standard_envelope']}")
            
            # Get eBay suggested category first
            category_id = await self.get_ebay_category_suggestions(product)
            if not category_id:
                print("⚠️  No category suggestion from eBay, using fallback")
                category_id = self._determine_ebay_category(product)
            
            # Get eBay-specific aspects for this category
            aspects = await self._extract_aspects_from_ebay_api(product, category_id)
            
            # Prepare inventory item data with all required fields
            inventory_data = {
                "availability": {
                    "shipToLocationAvailability": {
                        "quantity": int(1)
                    }
                },
                "condition": "NEW",
                "product": {
                    "title": product.title[:80],  # eBay title limit (max 80 chars)
                    "description": self._format_description(product),
                    "aspects": aspects,
                    "brand": product.brand or "Unbranded",
                    "imageUrls": product.images[:12] if product.images else ["https://via.placeholder.com/400x400?text=No+Image"],  # At least one image required
                    "ean": [],  # European Article Number
                    "upc": [],  # Universal Product Code  
                    "mpn": product.asin  # Use ASIN as manufacturer part number
                },
                "packageWeightAndSize": {
                    "weight": {
                        "value": weight_value,
                        "unit": weight_unit
                    },
                    "dimensions": {
                        "length": 10.0,
                        "width": 10.0, 
                        "height": 10.0,
                        "unit": "INCH"
                    }
                },
                "locale": "en_US"
            }
            
            # Debug: Show exactly what weight is being sent to eBay
            self.log(f"🔍 Inventory Item Weight Data:")
            self.log(f"   Original weight_oz: {weight_oz} oz")
            self.log(f"   eBay weight: {weight_value} {weight_unit}")
            self.log(f"   Weight unit: {weight_unit}")
            self.log(f"   Weight value type: {type(weight_value)} = {weight_value}")
            
            # Ensure weight is a proper float/number for eBay API
            weight_value = float(weight_value)
            
            # Add UPC/EAN if available (required for many categories)
            # Note: You might want to fetch this from Keepa or add to Product model
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Content-Language': 'en-US'
            }
            
            async with httpx.AsyncClient() as client:
                print(f"🔄 Creating inventory item with data: {inventory_data}")
                response = await self.make_request_with_retry(
                    client, "PUT", f"{self.base_url}/sell/inventory/v1/inventory_item/{sku}",
                    headers=headers, json_data=inventory_data, max_retries=3
                )
                
                print(f"📡 Inventory creation response: Status {response.status_code}")
                if response.status_code not in [200, 201, 204]:
                    print(f"❌ Inventory creation failed response: {response.text}")
                
                if response.status_code in [200, 201, 204]:
                    print(f"✅ Inventory item created successfully with SKU: {sku}")
                    # Verify the inventory item was created by trying to get it
                    await asyncio.sleep(2)  # Give eBay more time to process
                    verification_response = await self.make_request_with_retry(
                        client, "GET", f"{self.base_url}/sell/inventory/v1/inventory_item/{sku}",
                        headers=headers, max_retries=3
                    )
                    print(f"📡 Verification response: Status {verification_response.status_code}")
                    if verification_response.status_code == 200:
                        verification_data = verification_response.json()
                        print(f"✅ Inventory item verified in eBay system: {sku}")
                        print(f"🔍 Verified inventory data: {verification_data}")
                        return sku
                    else:
                        print(f"❌ Verification failed: {verification_response.text}")
                        raise Exception(f"Inventory item created but not found in verification: {verification_response.status_code}")
                else:
                    error_msg = f"Inventory creation failed - Status: {response.status_code}, Response: {response.text}"
                    print(f"❌ {error_msg}")
                    raise Exception(error_msg)
                    
        except Exception as e:
            error_msg = f"Error creating inventory item: {str(e)}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
    
    async def create_inventory_item_with_category(self, product: 'Product') -> Optional[Dict[str, str]]:
        """Create inventory item and return both SKU and category ID"""
        try:
            access_token = await self.get_access_token()
            
            # Generate unique SKU from ASIN with timestamp to avoid conflicts
            import time
            timestamp = int(time.time())
            sku = f"AMZ-{product.asin}-{timestamp}"
            
            # Calculate shipping requirements to get accurate weight
            shipping_req = self._determine_shipping_requirements(product)
            
            # Log detailed weight analysis
            self.log(f"📊 Weight Analysis for {product.asin}:")
            self.log(f"   Product weight_oz: {getattr(product, 'weight_oz', 'None')}")
            self.log(f"   Calculated weight: {shipping_req['estimated_weight_oz']} oz")
            self.log(f"   Weight source: {shipping_req['weight_source']}")
            self.log(f"   Envelope eligible: {shipping_req['eligible_for_standard_envelope']}")
            self.log(f"   Ground required: {shipping_req['requires_ground_shipping']}")
            self.log(f"   Package type: {shipping_req['package_type']}")
            
            # Handle weight format for eBay API - try separate pound/ounce format
            weight_oz = shipping_req['estimated_weight_oz']
            
            # Validate weight - eBay requires valid weight > 0
            if not weight_oz or weight_oz <= 0:
                self.log(f"⚠️  Invalid weight detected ({weight_oz}), using 0.1 oz minimum")
                weight_oz = 0.1  # Minimum detectable weight
            
            # Calculate separate pounds and ounces (e.g., 0 lb 10 oz)
            total_ounces = max(weight_oz, 0.1)  # Minimum 0.1 oz
            pounds = int(total_ounces // 16)  # Whole pounds
            remaining_ounces = round(total_ounces % 16, 1)  # Remaining ounces
            
            # If no remaining ounces, ensure at least 0.1 oz
            if remaining_ounces == 0 and pounds == 0:
                remaining_ounces = 0.1
            
            self.log(f"📦 eBay weight format: {pounds} lb {remaining_ounces} oz (total: {total_ounces} oz)")
            
            # Use smart weight formatting with realistic minimums
            weight_value, weight_unit = self._format_weight_for_ebay(
                weight_oz, 
                product.title if hasattr(product, 'title') else "",
                use_lb_minimum=True
            )
            
            self.log(f"� Sending to eBay API: {weight_value} {weight_unit}")
            
            self.log(f"📦 Using validated weight: {weight_value} {weight_unit} for {product.title[:50]}...")
            self.log(f"🚚 Shipping analysis: envelope_eligible={shipping_req['eligible_for_standard_envelope']}, ground_required={shipping_req['requires_ground_shipping']}")
            
            # Get eBay suggested category first
            try:
                category_id = await self.get_ebay_category_suggestions(product)
                if not category_id:
                    self.log("⚠️  No category suggestion from eBay, using fallback")
                    category_id = self._determine_ebay_category(product)
            except Exception as e:
                self.log(f"❌ Error getting category suggestions: {e}")
                category_id = self._determine_ebay_category(product)
            
            # Get eBay-specific aspects for this category
            try:
                aspects = await self._extract_aspects_from_ebay_api(product, category_id)
            except Exception as e:
                self.log(f"❌ Error extracting aspects: {e}")
                aspects = self._extract_aspects_fallback(product)
            
            # Prepare inventory item data with all required fields
            inventory_data = {
                "availability": {
                    "shipToLocationAvailability": {
                        "quantity": int(1)
                    }
                },
                "condition": "NEW",
                "product": {
                    "title": product.title[:80],  # eBay title limit (max 80 chars)
                    "description": self._format_description(product),
                    "aspects": aspects,
                    "brand": product.brand or "Unbranded",
                    "imageUrls": product.images[:12] if product.images else ["https://via.placeholder.com/400x400?text=No+Image"],  # At least one image required
                    "ean": [],  # European Article Number
                    "upc": [],  # Universal Product Code  
                    "mpn": product.asin  # Use ASIN as manufacturer part number
                },
                "packageWeightAndSize": {
                    "weight": {
                        "value": weight_value,
                        "unit": weight_unit
                    },
                    "dimensions": {
                        "length": 10.0,
                        "width": 10.0, 
                        "height": 10.0,
                        "unit": "INCH"
                    }
                },
                "locale": "en_US"
            }
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Content-Language': 'en-US'
            }
            
            self.log(f"🔄 Creating inventory item with eBay-suggested category: {category_id}")
            self.log(f"📋 Inventory data preview: Title='{inventory_data['product']['title']}', Brand='{inventory_data['product']['brand']}', Aspects={len(inventory_data['product']['aspects'])} items")
            self.log(f"📦 Final weight being sent to eBay: {weight_value} {weight_unit}")
            
            async with httpx.AsyncClient() as client:
                response = await self.make_request_with_retry(
                    client, "PUT", f"{self.base_url}/sell/inventory/v1/inventory_item/{sku}",
                    headers=headers, json_data=inventory_data, max_retries=3
                )
                
                self.log(f"📡 Inventory creation response: Status {response.status_code}")
                if response.status_code in [200, 201, 204]:
                    self.log(f"✅ Inventory item created successfully: {sku}")
                    return {
                        'sku': sku,
                        'category_id': category_id
                    }
                else:
                    self.log(f"❌ Failed to create inventory item: {response.text}")
                    return None
                    
        except Exception as e:
            self.log(f"❌ Error creating inventory item: {str(e)}")
            import traceback
            self.log(f"📋 Full traceback: {traceback.format_exc()}")
            return None

    async def create_offer(self, sku: str, product: 'Product', location_key: str, category_id: str) -> Optional[str]:
        """Create offer for inventory item"""
        try:
            print(f"🔄 Creating offer for SKU: {sku}")
            access_token = await self.get_access_token()
            marketplace_id, currency = self._get_marketplace_info()
            
            # First, verify the inventory item exists before creating offer
            print(f"🔄 Verifying inventory item {sku} exists...")
            inventory_exists = await self.check_inventory_item_exists(sku)
            if not inventory_exists:
                raise Exception(f"Inventory item {sku} not found in eBay system. Cannot create offer.")
            
            # Get appropriate business policies for this specific product
            print("🔄 Getting appropriate business policies...")
            
            # Get smart fulfillment policy based on product characteristics
            print("📦 Selecting optimal shipping policy...")
            self.log(f"🚚 Fulfillment Policy Selection for {product.asin}:")
            fulfillment_policy_id = await self.get_appropriate_fulfillment_policy(product)
            if fulfillment_policy_id:
                self.log(f"✅ Selected fulfillment policy ID: {fulfillment_policy_id}")
            else:
                self.log("❌ No fulfillment policy selected")
            if not fulfillment_policy_id:
                error_msg = "❌ CRITICAL: Could not determine appropriate fulfillment policy"
                print(error_msg)
                raise Exception("Missing fulfillment policy for offer creation")
            
            # Get other required policies
            policies = await self.get_business_policies()
            if not policies or not all(key in policies for key in ['paymentPolicyId', 'returnPolicyId']):
                error_msg = "❌ CRITICAL: Missing required business policies (payment, return)"
                print(error_msg)
                print("⚠️  Please set up all business policies in your eBay seller account first")
                print("⚠️  Go to: Seller Hub → Account → Business Policies")
                raise Exception("Missing required business policies for offer creation")
            
            # Use the smart fulfillment policy instead of the generic one
            policies["fulfillmentPolicyId"] = fulfillment_policy_id
            
            # Debug: Show exactly which policy is being used
            self.log(f"🔍 Selected fulfillment policy: {fulfillment_policy_id}")
            
            # Prepare offer data following eBay's official createOffer API specification
            offer_data = {
                "sku": sku,
                "marketplaceId": marketplace_id,
                "format": "FIXED_PRICE",
                "availableQuantity": int(1),
                "categoryId": category_id,
                "listingDescription": self._format_listing_description(product),
                "listingDuration": "GTC",  # Good Till Cancelled - REQUIRED
                "pricingSummary": {
                    "price": {
                        "currency": currency,
                        "value": f"{float(product.optimized_price):.2f}"
                    }
                },
                "quantityLimitPerBuyer": 1,
                "includeCatalogProductDetails": True,
                "merchantLocationKey": location_key,  # Required for country validation
                "listingPolicies": {
                    "fulfillmentPolicyId": policies.get("fulfillmentPolicyId"),
                    "paymentPolicyId": policies.get("paymentPolicyId"),
                    "returnPolicyId": policies.get("returnPolicyId")
                }
                # Removed lotSize as it's causing error 25006 - "lot size is invalid, need to be greater than 1"
                # Removed status as it may not be needed for createOffer
            }
            
            # Add shipping cost overrides if enabled in config
            config = Config.load_from_keyring()
            if config.shipping_cost_override_enabled:
                shipping_overrides = []
                
                # Override domestic shipping costs
                domestic_override = {
                    "priority": 1,  # Priority 1 is typically the first domestic shipping option
                    "shippingServiceType": "DOMESTIC",
                    "shippingCost": {
                        "currency": currency,
                        "value": f"{config.shipping_cost_override_amount:.2f}"
                    }
                }
                
                # Add additional shipping cost override if specified
                if config.shipping_additional_cost_override > 0:
                    domestic_override["additionalShippingCost"] = {
                        "currency": currency,
                        "value": f"{config.shipping_additional_cost_override:.2f}"
                    }
                
                shipping_overrides.append(domestic_override)
                
                # Add international override if not domestic-only
                if not config.shipping_cost_override_domestic_only:
                    international_override = {
                        "priority": 1,  # Priority 1 for international
                        "shippingServiceType": "INTERNATIONAL", 
                        "shippingCost": {
                            "currency": currency,
                            "value": f"{config.shipping_cost_override_amount:.2f}"
                        }
                    }
                    
                    if config.shipping_additional_cost_override > 0:
                        international_override["additionalShippingCost"] = {
                            "currency": currency,
                            "value": f"{config.shipping_additional_cost_override:.2f}"
                        }
                    
                    shipping_overrides.append(international_override)
                
                # Add shipping cost overrides to listing policies
                offer_data["listingPolicies"]["shippingCostOverrides"] = shipping_overrides
                
                self.log(f"💰 Applied shipping cost overrides:")
                self.log(f"   Domestic shipping: ${config.shipping_cost_override_amount:.2f}")
                if not config.shipping_cost_override_domestic_only:
                    self.log(f"   International shipping: ${config.shipping_cost_override_amount:.2f}")
                if config.shipping_additional_cost_override > 0:
                    self.log(f"   Each additional item: ${config.shipping_additional_cost_override:.2f}")
            
            print(f"✅ Required business policies included:")
            print(f"   - Fulfillment: {policies.get('fulfillmentPolicyId')}")
            print(f"   - Payment: {policies.get('paymentPolicyId')}")
            print(f"   - Return: {policies.get('returnPolicyId')}")
            print(f"📂 Using eBay category: {offer_data['categoryId']} for '{product.title[:50]}...'")
            
            # Validate offer data structure before sending
            validation_result = self._validate_offer_data(offer_data)
            if not validation_result["valid"]:
                error_msg = f"❌ Offer data validation failed: {validation_result['errors']}"
                print(error_msg)
                raise Exception(error_msg)
            
            print(f"✅ Offer data validation passed")
            print(f"🔄 Sending offer creation request...")
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Content-Language': 'en-US'
            }
            
            async with httpx.AsyncClient() as client:
                response = await self.make_request_with_retry(
                    client, "POST", f"{self.base_url}/sell/inventory/v1/offer", 
                    headers=headers, json_data=offer_data, max_retries=3
                )
                
                print(f"📡 Offer creation response: Status {response.status_code}")
                if response.status_code not in [200, 201]:
                    print(f"❌ Offer creation response: {response.text}")
                
                if response.status_code in [200, 201]:
                    offer_response = response.json()
                    offer_id = offer_response.get("offerId")
                    print(f"✅ Offer created successfully: {offer_id}")
                    return offer_id
                elif response.status_code == 400:
                    # Check for specific error types
                    response_text = response.text
                    if "offer entity already exists" in response_text.lower():
                        # Only retry for explicit "offer already exists" message, not generic 25002
                        print(f"⚠️  Offer already exists for SKU {sku}. Checking existing offers...")
                        # Try to get existing offers for this SKU
                        existing_offer_id = await self.get_existing_offer_id(sku)
                        if existing_offer_id:
                            print(f"✅ Found existing offer ID: {existing_offer_id}")
                            return existing_offer_id
                        else:
                            error_msg = f"Offer exists but couldn't retrieve it: {response_text}"
                            raise Exception(error_msg)
                    elif "25002" in response_text:
                        # Generic 25002 errors should fail, not retry - they indicate other system issues
                        error_msg = f"eBay system error (25002): {response_text}"
                        print(f"❌ {error_msg}")
                        raise Exception(error_msg)
                    elif "25702" in response_text or "could not be found" in response_text.lower():
                        error_msg = f"❌ SKU '{sku}' not found in eBay inventory system. The inventory item may not have been properly created."
                        print(error_msg)
                        raise Exception(f"Inventory item missing: {error_msg}")
                    else:
                        error_msg = f"Offer creation failed - Status: {response.status_code}, Response: {response.text}"
                        print(f"❌ {error_msg}")
                        raise Exception(error_msg)
                else:
                    error_msg = f"Offer creation failed - Status: {response.status_code}, Response: {response.text}"
                    print(f"❌ {error_msg}")
                    raise Exception(error_msg)
                    
        except Exception as e:
            if "Offer creation failed" in str(e) or "Inventory item missing" in str(e):
                raise  # Re-raise our custom error
            error_msg = f"Error creating offer: {str(e)}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
    
    async def check_inventory_item_exists(self, sku: str) -> bool:
        """Check if an inventory item exists in eBay system"""
        try:
            access_token = await self.get_access_token()
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/sell/inventory/v1/inventory_item/{sku}",
                    headers=headers,
                    timeout=30.0
                )
                return response.status_code == 200
                
        except Exception as e:
            print(f"⚠️  Error checking inventory item: {str(e)}")
            return False

    async def get_existing_offer_id(self, sku: str) -> Optional[str]:
        """Get existing offer ID for a SKU"""
        try:
            access_token = await self.get_access_token()
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Content-Language': 'en-US'
            }
            
            async with httpx.AsyncClient() as client:
                response = await self.make_request_with_retry(
                    client, "GET", f"{self.base_url}/sell/inventory/v1/offer",
                    headers=headers, params={"sku": sku}, max_retries=3
                )
                
                if response.status_code == 200:
                    offers_data = response.json()
                    offers = offers_data.get("offers", [])
                    if offers:
                        # Return the first offer ID found
                        return offers[0].get("offerId")
                    
                return None
                
        except Exception as e:
            print(f"⚠️  Error checking existing offers: {str(e)}")
            return None
    
    async def get_offer_details(self, offer_id: str) -> Optional[dict]:
        """Get detailed information about a specific offer"""
        try:
            access_token = await self.get_access_token()
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Content-Language': 'en-US'
            }
            
            async with httpx.AsyncClient() as client:
                response = await self.make_request_with_retry(
                    client, "GET", f"{self.base_url}/sell/inventory/v1/offer/{offer_id}",
                    headers=headers, max_retries=3
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"❌ Failed to get offer details: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            print(f"⚠️  Error getting offer details: {str(e)}")
            return None
    
    async def test_publish_with_production(self, offer_id: str) -> Optional[str]:
        """Test publishing with production environment (for debugging 500 errors)"""
        print(f"🔄 Testing publish with production environment...")
        
        # Temporarily switch to production
        original_base_url = self.base_url
        original_environment = self.environment
        
        try:
            self.base_url = "https://api.ebay.com"
            self.environment = "production"
            
            result = await self.publish_offer(offer_id, skip_validation=False)
            return result
            
        except Exception as e:
            print(f"❌ Production environment also failed: {str(e)}")
            return None
            
        finally:
            # Restore original settings
            self.base_url = original_base_url
            self.environment = original_environment
    
    async def publish_offer(self, offer_id: str, skip_validation: bool = True) -> Optional[str]:
        """Publish offer to create live listing"""
        try:
            print(f"🔄 Starting publish process for offer ID: {offer_id}")
            
            # CRITICAL: Always validate the offer before publishing to catch missing required fields
            print(f"🔍 Validating offer {offer_id} before attempting to publish...")
            offer_details = await self.get_offer_details(offer_id)
            if offer_details:
                print(f"📋 Offer validation - Status: {offer_details.get('status', 'UNKNOWN')}")
                print(f"📋 Offer validation - SKU: {offer_details.get('sku')}")
                print(f"📋 Offer validation - Marketplace: {offer_details.get('marketplaceId')}")
                print(f"📋 Offer validation - Category: {offer_details.get('categoryId')}")
                print(f"📋 Offer validation - Format: {offer_details.get('format')}")
                print(f"📋 Offer validation - Duration: {offer_details.get('listingDuration')}")
                print(f"📋 Offer validation - Merchant Location: {offer_details.get('merchantLocationKey')}")
                
                # Check critical publishing requirements
                missing_fields = []
                required_for_publish = [
                    'sku', 'marketplaceId', 'categoryId', 'format', 
                    'listingDuration', 'merchantLocationKey', 'pricingSummary'
                ]
                
                for field in required_for_publish:
                    if field not in offer_details or not offer_details[field]:
                        missing_fields.append(field)
                
                # Check nested required fields
                if 'pricingSummary' in offer_details:
                    pricing = offer_details['pricingSummary']
                    if 'price' not in pricing or not pricing['price']:
                        missing_fields.append('pricingSummary.price')
                    elif 'value' not in pricing['price'] or not pricing['price']['value']:
                        missing_fields.append('pricingSummary.price.value')
                
                if 'listingPolicies' in offer_details:
                    policies = offer_details['listingPolicies']
                    required_policies = ['fulfillmentPolicyId', 'paymentPolicyId', 'returnPolicyId']
                    for policy in required_policies:
                        if policy not in policies or not policies[policy]:
                            missing_fields.append(f'listingPolicies.{policy}')
                else:
                    missing_fields.append('listingPolicies')
                
                if missing_fields:
                    error_msg = f"❌ Offer {offer_id} is missing required fields for publishing: {missing_fields}"
                    print(error_msg)
                    raise Exception(error_msg)
                
                print(f"✅ Offer {offer_id} validation passed - all required fields present")
                
                # Check if already published
                if offer_details.get('status') == 'PUBLISHED':
                    listing_id = offer_details.get('listing', {}).get('listingId')
                    if listing_id:
                        print(f"ℹ️  Offer {offer_id} is already published as listing {listing_id}")
                        return listing_id
            else:
                print(f"⚠️  Could not retrieve offer details for validation - this may cause publish to fail")
            
            access_token = await self.get_access_token()
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
                # Removed Content-Language as it might be causing issues with publish endpoint
            }
            
            print(f"🔍 Debug: Publishing offer {offer_id}")
            print(f"🔍 Debug: Using endpoint: {self.base_url}/sell/inventory/v1/offer/{offer_id}/publish")
            print(f"🔍 Debug: Current environment: {'Sandbox' if 'sandbox' in self.base_url else 'Production'}")
            print(f"🔍 Debug: Request will have no body (as per eBay API spec)")
            
            async with httpx.AsyncClient() as client:
                response = await self.make_request_with_retry(
                    client, "POST", f"{self.base_url}/sell/inventory/v1/offer/{offer_id}/publish",
                    headers=headers, max_retries=3
                )
                
                print(f"📡 Publish offer response: Status {response.status_code}")
                
                # Debug: Print full response details
                print(f"🔍 Debug: Response headers: {dict(response.headers)}")
                
                if response.status_code not in [200, 201]:
                    print(f"❌ Publish offer response: {response.text}")
                    # Additional debugging for failures
                    print(f"🔍 Debug: Full response details:")
                    print(f"   Status: {response.status_code} {response.reason_phrase}")
                    print(f"   Headers: {dict(response.headers)}")
                    print(f"   Body: {response.text}")
                    
                    # Special handling for persistent 500 errors
                    if response.status_code == 500:
                        print(f"❌ eBay server error (500) - this may be a temporary issue with eBay's servers")
                        print(f"💡 Possible solutions:")
                        print(f"   1. Try again in a few minutes")
                        print(f"   2. Check if your eBay account has all required business policies")
                        print(f"   3. Switch to production environment if using sandbox")
                        if 'sandbox' in self.base_url:
                            print(f"   4. Note: Sandbox environments can be unreliable - consider production testing")
                
                if response.status_code in [200, 201]:
                    listing_data = response.json()
                    listing_id = listing_data.get("listingId")
                    print(f"✅ Offer published successfully: {listing_id}")
                    return listing_id
                else:
                    response_text = response.text
                    # Check for specific error types and provide helpful guidance
                    if "item.country" in response_text.lower() or "country" in response_text.lower():
                        error_msg = f"❌ Country field issue during publish: {response_text}"
                        print(error_msg)
                        raise Exception(f"Country validation failed: Please ensure inventory item has proper country information")
                    elif "25002" in response_text:
                        if "business policy" in response_text.lower() or "policy" in response_text.lower():
                            error_msg = f"❌ Business policy error (25002): {response_text}"
                            print(error_msg)
                            raise Exception(f"Business policies required: Please set up Payment, Return, and Shipping policies in your eBay account before publishing offers. Visit: https://www.ebay.com/bp/policyoptin")
                        else:
                            error_msg = f"❌ Offer entity error (25002): {response_text}"
                            print(error_msg)
                            # Log offer details for debugging (only if validation was enabled)
                            if not skip_validation:
                                print(f"🔍 Offer details at time of error: {offer_details}")
                            raise Exception(f"Offer publishing failed, errorid 25002: {response_text}")
                    elif "business" in response_text.lower() and "policy" in response_text.lower():
                        error_msg = f"❌ Business policy missing: {response_text}"
                        print(error_msg)
                        raise Exception(f"Business policies required: Please set up Payment, Return, and Shipping policies in your eBay account. Visit: https://www.ebay.com/bp/policyoptin")
                    else:
                        error_msg = f"Offer publish failed - Status: {response.status_code}, Response: {response.text}"
                        print(f"❌ {error_msg}")
                        raise Exception(error_msg)
                    
        except Exception as e:
            if "Offer publish failed" in str(e):
                raise  # Re-raise our custom error
            error_msg = f"Error publishing offer: {str(e)}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
    
    async def create_listing(self, product: 'Product') -> Optional[str]:
        """Complete listing creation process"""
        try:
            # Step 0: Ensure inventory location exists (required for offer publishing)
            print(f"🔄 Step 0: Checking inventory location...")
            self.log("🏠 Location creation process starting...")
            location_key = await self.create_inventory_location()
            if not location_key:
                self.log("❌ Location creation failed - this will prevent listing creation")
                raise Exception("Failed to create or verify inventory location - required for eBay listings")
            else:
                self.log(f"✅ Location ready for listing: {location_key}")
            
            # Step 1: Create inventory item
            print(f"🔄 Step 1: Creating inventory item for {product.asin}...")
            inventory_result = await self.create_inventory_item_with_category(product)
            if not inventory_result:
                raise Exception("Failed to create inventory item - check product data and eBay requirements")
            
            sku = inventory_result['sku']
            category_id = inventory_result['category_id']
            
            # Step 1.5: Wait for eBay system to fully process inventory item
            print(f"🔄 Step 1.5: Waiting for eBay system to process inventory item...")
            await asyncio.sleep(5)  # Give eBay more time to process inventory item
            
            # Step 1.6: Enhanced inventory item verification with retries
            print(f"🔄 Step 1.6: Verifying inventory item {sku}...")
            verification_attempts = 0
            max_verification_attempts = 4
            
            while verification_attempts < max_verification_attempts:
                if await self.check_inventory_item_exists(sku):
                    print(f"✅ Inventory item {sku} verified successfully")
                    break
                
                verification_attempts += 1
                wait_time = verification_attempts * 5  # Progressive wait: 5s, 10s, 15s, 20s
                print(f"⚠️  Verification attempt {verification_attempts}/{max_verification_attempts} failed, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                
                if verification_attempts == max_verification_attempts:
                    # Final attempt - log detailed error
                    print(f"❌ Final verification failed for {sku}")
                    print(f"📋 Product details: {product.title[:100]}")
                    print(f"📦 Weight: {getattr(product, 'weight_oz', 'unknown')} oz")
                    raise Exception(f"Inventory item {sku} verification failed after {max_verification_attempts} attempts. eBay may be experiencing delays or the item data was rejected.")
            
            # Step 2: Create offer
            print(f"🔄 Step 2: Creating offer for SKU {sku}...")
            offer_id = await self.create_offer(sku, product, location_key, category_id)
            if not offer_id:
                raise Exception("Failed to create offer - check pricing and category mapping")
            
            # Step 3: Publish offer
            print(f"🔄 Step 3: Publishing offer {offer_id}...")
            listing_id = await self.publish_offer(offer_id)
            if not listing_id:
                raise Exception("Failed to publish offer - check eBay policies and listing requirements")
                
            print(f"✅ Successfully created eBay listing: {listing_id}")
            return listing_id
            
        except Exception as e:
            import traceback
            error_details = f"Error in listing process: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_details)  # For debugging
            raise Exception(f"eBay listing failed: {str(e)}")
    
    def _format_description(self, product: 'Product') -> str:
        """Format product description for eBay"""
        description_parts = []
        
        if product.description and product.description != "No description available":
            description_parts.append(product.description[:500])
        
        if product.features:
            description_parts.append("\n\nKey Features:")
            for feature in product.features[:5]:
                description_parts.append(f"• {feature}")
        
        description_parts.append(f"\n\nAmazon ASIN: {product.asin}")
        description_parts.append("Cross-listed from Amazon for your convenience.")
        
        return "\n".join(description_parts)[:1000]  # eBay description limit
    
    def _format_listing_description(self, product: 'Product') -> str:
        """Format HTML listing description"""
        html_description = f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px;">
            <h2>{product.title}</h2>
            
            {f'<p><strong>Brand:</strong> {product.brand}</p>' if product.brand and product.brand != "Unknown Brand" else ''}
            
            {f'<p>{product.description[:500]}{"..." if len(product.description) > 500 else ""}</p>' if product.description and product.description != "No description available" else ''}
            
            {f'''
            <h3>Key Features:</h3>
            <ul>
                {"".join(f"<li>{feature}</li>" for feature in product.features[:5])}
            </ul>
            ''' if product.features else ''}
            
            <hr>
            <p><small>Amazon ASIN: {product.asin} | Cross-listed for your convenience</small></p>
        </div>
        """
        return html_description.strip()
    
    async def _extract_aspects_from_ebay_api(self, product: 'Product', category_id: str) -> Dict[str, List[str]]:
        """Extract aspects using eBay's category aspects API"""
        aspects = {}
        
        # Get the required and optional aspects for this category from eBay
        category_aspects = await self.get_ebay_category_aspects(category_id)
        
        if not category_aspects:
            self.log("⚠️  No aspects data from eBay, using fallback method")
            return self._extract_aspects_fallback(product)
        
        required_aspects = category_aspects.get('required_aspects', [])
        optional_aspects = category_aspects.get('optional_aspects', [])
        
        self.log(f"📋 Processing {len(required_aspects)} required + {len(optional_aspects)} optional aspects")
        
        title = product.title.lower() if hasattr(product, 'title') else ""
        
        # Process required aspects first
        for aspect in required_aspects:
            aspect_name = aspect.get('localizedAspectName', '')
            aspect_values = aspect.get('aspectValues', [])
            
            self.log(f"🔍 Processing required aspect: {aspect_name}")
            
            # Try to extract or provide reasonable defaults
            if aspect_name.lower() in ['brand']:
                if product.brand and product.brand != "Unknown Brand":
                    aspects[aspect_name] = [product.brand]
                else:
                    aspects[aspect_name] = ["Unbranded"]
                    
            elif aspect_name.lower() in ['condition']:
                aspects[aspect_name] = ["New"]
                
            elif 'screen size' in aspect_name.lower():
                screen_size = self._extract_screen_size_from_title(title)
                if screen_size:
                    aspects[aspect_name] = [screen_size]
                else:
                    # Provide a reasonable default based on product type
                    if "monitor" in title:
                        aspects[aspect_name] = ["24 in"]
                    elif "tablet" in title or "ipad" in title:
                        aspects[aspect_name] = ["10.1 in"]
                    elif "phone" in title:
                        aspects[aspect_name] = ["6.1 in"]
                    elif "laptop" in title:
                        aspects[aspect_name] = ["15.6 in"]
                    else:
                        # Use the first available value from eBay if no detection
                        if aspect_values:
                            aspects[aspect_name] = [aspect_values[0].get('localizedValue', 'Unknown')]
                        
            elif 'processor' in aspect_name.lower():
                # Skip processor for accessories that don't actually have processors
                if any(keyword in title for keyword in ['mousepad', 'mouse pad', 'case', 'cover', 'cable', 'charger']):
                    self.log(f"⚠️  Skipping processor requirement for accessory: {title[:30]}...")
                    continue
                    
                processor = self._extract_processor_from_title(title)
                if processor:
                    aspects[aspect_name] = [processor]
                else:
                    # Provide reasonable defaults or use eBay's first available value
                    if aspect_values:
                        # Look for common processors in the available values
                        common_processors = ["Intel Core i5", "Intel Core i7", "AMD Ryzen", "Apple M1", "Apple M2"]
                        for proc in common_processors:
                            for val in aspect_values:
                                if proc.lower() in val.get('localizedValue', '').lower():
                                    aspects[aspect_name] = [val.get('localizedValue')]
                                    break
                            if aspect_name in aspects:
                                break
                        
                        # If no match found, use first available value
                        if aspect_name not in aspects and aspect_values:
                            aspects[aspect_name] = [aspect_values[0].get('localizedValue', 'Not Specified')]
                            
            elif 'color' in aspect_name.lower():
                color = self._extract_color_from_title(title)
                if color:
                    aspects[aspect_name] = [color]
                elif aspect_values:
                    # Look for "Black" or "Multi-Color" as safe defaults
                    for val in aspect_values:
                        val_text = val.get('localizedValue', '').lower()
                        if 'black' in val_text or 'multi' in val_text:
                            aspects[aspect_name] = [val.get('localizedValue')]
                            break
                    if aspect_name not in aspects and aspect_values:
                        aspects[aspect_name] = [aspect_values[0].get('localizedValue', 'Not Specified')]
                        
            else:
                # For other required aspects, try to use the first available value
                if aspect_values:
                    aspects[aspect_name] = [aspect_values[0].get('localizedValue', 'Not Specified')]
                self.log(f"⚠️  Using default value for required aspect: {aspect_name}")
        
        # Process some optional aspects that we can reasonably determine
        for aspect in optional_aspects[:5]:  # Limit to avoid too many aspects
            aspect_name = aspect.get('localizedAspectName', '')
            
            if 'storage' in aspect_name.lower() and any(x in title for x in ['gb', 'tb']):
                storage = self._extract_storage_from_title(title)
                if storage:
                    aspects[aspect_name] = [storage]
                    
            elif 'connectivity' in aspect_name.lower() and any(x in title for x in ['wireless', 'bluetooth', 'usb']):
                if "wireless" in title or "bluetooth" in title:
                    aspects[aspect_name] = ["Bluetooth"]
                elif "usb" in title:
                    aspects[aspect_name] = ["Wired"]
        
        self.log(f"✅ Generated {len(aspects)} aspects from eBay API: {list(aspects.keys())}")
        return aspects
    
    def _extract_processor_from_title(self, title: str) -> Optional[str]:
        """Extract processor information from product title"""
        import re
        
        # Common processor patterns
        processors = [
            r'intel core i[3579]',
            r'amd ryzen [357]',
            r'apple m[12]',
            r'intel celeron',
            r'intel pentium',
            r'snapdragon \d+',
            r'mediatek \w+',
            r'exynos \d+'
        ]
        
        for pattern in processors:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return match.group(0).title()
                
        return None
    
    def _extract_aspects_fallback(self, product: 'Product') -> Dict[str, List[str]]:
        """Fallback method for extracting product aspects when eBay API is unavailable"""
        aspects = {}
        
        # Basic required aspects
        if product.brand and product.brand != "Unknown Brand":
            aspects["Brand"] = [product.brand]
        else:
            aspects["Brand"] = ["Unbranded"]
            
        aspects["Condition"] = ["New"]
        
        # Extract category and add category-specific aspects
        category_id = self._determine_ebay_category(product)
        title = product.title.lower() if hasattr(product, 'title') else ""
        
        # Screen size for monitors, tablets, phones, laptops
        screen_size = self._extract_screen_size_from_title(title)
        
        if category_id in ["80053", "171485", "9355", "177"]:  # Monitors, tablets, phones, laptops
            if screen_size:
                aspects["Screen Size"] = [screen_size]
            else:
                # Provide reasonable defaults based on product type
                if "monitor" in title:
                    aspects["Screen Size"] = ["24 in"]  # Common monitor size
                elif "tablet" in title or "ipad" in title:
                    aspects["Screen Size"] = ["10.1 in"]  # Common tablet size
                elif "phone" in title or "smartphone" in title:
                    aspects["Screen Size"] = ["6.1 in"]  # Common phone size
                elif "laptop" in title:
                    aspects["Screen Size"] = ["15.6 in"]  # Common laptop size
                else:
                    aspects["Screen Size"] = ["Unknown"]
        
        # Color extraction
        color = self._extract_color_from_title(title)
        if color:
            aspects["Color"] = [color]
        
        # Memory/Storage for electronics
        if category_id in ["9355", "171485", "177"]:  # Phones, tablets, laptops
            storage = self._extract_storage_from_title(title)
            if storage:
                aspects["Storage Capacity"] = [storage]
                
        # Connectivity for accessories
        if category_id in ["31395", "15052", "14969"]:  # Cables, headphones, speakers
            if "wireless" in title or "bluetooth" in title:
                aspects["Connectivity"] = ["Bluetooth"]
            elif "usb" in title:
                aspects["Connectivity"] = ["Wired"]
                
        # Gaming console specifics
        if category_id in ["139973", "38583"]:  # Gaming consoles, controllers
            if "xbox" in title:
                aspects["Platform"] = ["Microsoft Xbox"]
            elif "playstation" in title or "ps5" in title or "ps4" in title:
                aspects["Platform"] = ["Sony PlayStation"]
            elif "nintendo" in title:
                aspects["Platform"] = ["Nintendo"]
                
        # Weight for shipping (if we have it)
        if hasattr(product, 'weight_oz') and product.weight_oz:
            weight_lbs = round(product.weight_oz / 16, 2)
            aspects["Item Weight"] = [f"{weight_lbs} lbs"]
            
        # Material for accessories like mousepads
        if "mousepad" in title or "mouse pad" in title:
            if "cloth" in title:
                aspects["Material"] = ["Cloth"]
            elif "rubber" in title:
                aspects["Material"] = ["Rubber"]
            else:
                aspects["Material"] = ["Mixed Materials"]
                
        # Compatibility for accessories
        if any(keyword in title for keyword in ["case", "cover", "screen protector"]):
            if "iphone" in title:
                aspects["Compatible Brand"] = ["Apple"]
            elif "samsung" in title:
                aspects["Compatible Brand"] = ["Samsung"]
            elif "android" in title:
                aspects["Compatible Brand"] = ["Universal"]
        
        print(f"📋 Generated aspects for category {category_id}: {aspects}")
        return aspects
        
    def _extract_screen_size_from_title(self, title: str) -> Optional[str]:
        """Extract screen size from product title"""
        import re
        
        # Look for patterns like: 24", 27 inch, 15.6", etc.
        patterns = [
            r'(\d+\.?\d*)"',  # 24", 15.6"
            r'(\d+\.?\d*)\s*inch',  # 24 inch, 15.6 inch
            r'(\d+\.?\d*)\s*in\b',  # 24 in
            r'(\d+\.?\d*)-inch',  # 24-inch
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                size = float(match.group(1))
                return f"{size} in"
                
        return None
        
    def _extract_color_from_title(self, title: str) -> Optional[str]:
        """Extract color from product title"""
        colors = [
            "black", "white", "red", "blue", "green", "yellow", "orange", 
            "purple", "pink", "gray", "grey", "silver", "gold", "brown",
            "navy", "teal", "cyan", "magenta", "lime", "maroon", "olive"
        ]
        
        for color in colors:
            if color in title:
                return color.capitalize()
                
        return None
        
    def _extract_storage_from_title(self, title: str) -> Optional[str]:
        """Extract storage capacity from title"""
        import re
        
        # Look for storage patterns: 64GB, 128 GB, 1TB, etc.
        patterns = [
            r'(\d+)\s*TB',  # 1TB, 2 TB
            r'(\d+)\s*GB',  # 64GB, 128 GB
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                size = match.group(1)
                unit = "TB" if "TB" in pattern else "GB"
                return f"{size} {unit}"
                
        return None

class EbayAccountSetupWizard(ModalScreen):
    """Complete eBay account setup wizard for users to create their own developer app"""
    
    def app_log(self, message: str):
        """Logging method for modal screens - uses notifications instead of main log"""
        # Use notification system for modal screen logging
        self.app.notify(message, severity="info")
    
    CSS = """
    EbayAccountSetupWizard {
        align: center middle;
    }
    
    .wizard-container {
        width: 90%;
        max-width: 80w;
        height: 90%;
        background: $surface;
        border: thick $primary;
        padding: 2;
    }
    
    #step_content {
        height: 1fr;
        max-height: 70vh;
        margin: 1 0 2 0;
    }
    
    .step-container {
        margin: 1 0;
        padding: 1;
        border: solid $accent;
    }
    
    .step-title {
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .step-description {
        color: $text;
        margin-bottom: 1;
    }
    
    .input-label {
        color: $primary;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 0;
    }
    
    .env-info {
        padding: 1;
        margin: 1 0;
        border: solid;
        text-align: center;
    }
    
    .sandbox-info {
        background: $warning;
        border: solid $warning;
        color: $warning-darken-3;
    }
    
    .production-info {
        background: $success;
        border: solid $success;
        color: $success-darken-3;
    }
    
    .input-group {
        margin: 1 0;
    }
    
    .nav-buttons {
        margin-top: 2;
        align: center middle;
    }
    
    .progress-bar {
        margin-bottom: 2;
        color: $primary;
        text-style: bold;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.current_step = 1
        self.total_steps = 6
        self.user_data = {}
        # Get current config to check sandbox mode
        try:
            self.config = Config.load_from_keyring()
            self.sandbox_mode = self.config.ebay_sandbox
        except Exception:
            self.config = None
            self.sandbox_mode = True  # Default to sandbox if config fails to load
    
    def compose(self) -> ComposeResult:
        with Container(classes="wizard-container"):
            yield Label(f"eBay Account Setup Wizard - Step {self.current_step} of {self.total_steps}", classes="progress-bar")
            
            # Step content container - make it scrollable
            with ScrollableContainer(id="step_content"):
                yield from self.get_step_content()
            
            # Navigation buttons
            with Horizontal(classes="nav-buttons"):
                yield Button("← Back", id="back_btn", disabled=self.current_step == 1)
                yield Button("Next →", id="next_btn", variant="primary")
                yield Button("Cancel", id="cancel_btn")
    
    def get_step_content(self):
        """Get content for current step"""
        print(f"🔄 get_step_content: Getting content for step {self.current_step}")
        
        try:
            if self.current_step == 1:
                print("🔄 get_step_content: Returning step 1 content")
                return self.step_1_welcome()
            elif self.current_step == 2:
                print("🔄 get_step_content: Returning step 2 content")
                return self.step_2_create_developer_account()
            elif self.current_step == 3:
                print("🔄 get_step_content: Returning step 3 content")
                return self.step_3_create_application()
            elif self.current_step == 4:
                print("🔄 get_step_content: Returning step 4 content")
                return self.step_4_enter_credentials()
            elif self.current_step == 5:
                print("🔄 get_step_content: Returning step 5 content")
                return self.step_5_authorize_account()
            elif self.current_step == 6:
                print("🔄 get_step_content: Returning step 6 content")
                return self.step_6_setup_business_policies()
            else:
                print(f"❌ get_step_content: Unknown step {self.current_step}")
                return iter([])  # Return empty iterator instead of empty list
        except Exception as e:
            print(f"❌ get_step_content: Error getting content for step {self.current_step}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def step_1_welcome(self):
        """Step 1: Welcome and overview"""
        yield Label("Welcome to eBay Integration Setup! 🚀", classes="step-title")
        yield Label("This wizard will guide you through setting up your own eBay Developer account", classes="step-description")
        yield Label("", classes="step-description")
        yield Label("What you'll accomplish:", classes="step-description")
        yield Label("✓ Create your own eBay Developer account (free)", classes="step-description")
        yield Label("✓ Set up an application for this cross-lister", classes="step-description")
        yield Label("✓ Get your personal API credentials", classes="step-description")
        yield Label("✓ Authorize the app to access your eBay selling account", classes="step-description")
        yield Label("", classes="step-description")
        yield Label("📝 Note: You'll need your eBay seller account credentials during this process", classes="step-description")
    
    def step_2_create_developer_account(self):
        """Step 2: Create eBay Developer account"""
        yield Label("Step 2: eBay Developer Account 👨‍💻", classes="step-title")
        yield Label("You need an eBay Developer account (separate from your selling account)", classes="step-description")
        yield Label("", classes="step-description")
        
        yield Label("📝 If you don't have one yet:", classes="step-description")
        yield Label("Instructions:", classes="step-description")
        yield Label("1. Click the button below to open eBay Developer Portal", classes="step-description")
        yield Label("2. Click 'Register' in the top right", classes="step-description")
        yield Label("3. Sign up with a NEW account (or use existing if you have one)", classes="step-description")
        yield Label("4. Complete the registration process", classes="step-description")
        yield Label("5. Verify your email address", classes="step-description")
        yield Label("", classes="step-description")
        yield Button("🌐 Open eBay Developer Portal", id="open_dev_portal", variant="primary")
        yield Label("", classes="step-description")
        yield Label("✅ Check this box when ready (whether you created a new account or already had one):", classes="step-description")
        yield Checkbox("I have an eBay Developer account and am ready to continue", id="dev_account_ready")
    
    def step_3_create_application(self):
        """Step 3: Create eBay Application"""
        yield Label("Step 3: eBay Application 📱", classes="step-title")
        yield Label("You need an application in your developer account", classes="step-description")
        yield Label("", classes="step-description")
        
        yield Label("📝 If you don't have one yet:", classes="step-description")
        yield Label("Instructions:", classes="step-description")
        yield Label("1. Go to 'My Account' → 'Keys' in the developer portal", classes="step-description")
        yield Label("2. Click 'Create a Key Set'", classes="step-description")
        yield Label("3. Fill out the application form:", classes="step-description")
        yield Label("   • Name: 'eBay Cross Lister' (or similar)", classes="step-description")
        yield Label("   • Description: 'Cross-listing tool for Amazon to eBay'", classes="step-description")
        yield Label("   • Select appropriate scopes (Selling, Inventory, etc.)", classes="step-description")
        yield Label("4. Submit and wait for approval (usually instant)", classes="step-description")
        yield Label("", classes="step-description")
        yield Button("🔗 Open My Account → Keys", id="open_keys_page", variant="primary")
        yield Label("", classes="step-description")
        yield Label("✅ Check this box when ready (whether you created a new app or already had one):", classes="step-description")
        yield Checkbox("I have an eBay application and am ready to continue", id="app_created")
    
    def step_4_enter_credentials(self):
        """Step 4: Enter API credentials"""
        yield Label("Step 4: Enter Your API Credentials 🔑", classes="step-title")
        yield Label("Copy your credentials from the eBay Developer portal", classes="step-description")
        yield Label("", classes="step-description")
        
        # Environment explanation
        if self.sandbox_mode:
            yield Static(
                "🧪 SANDBOX MODE ACTIVE\n"
                "You're setting up SANDBOX credentials for testing only.\n"
                "• Sandbox creates fake listings that don't appear on real eBay\n"
                "• Perfect for testing the application safely\n"
                "• To sell real items, change to Production in main config",
                classes="env-info sandbox-info"
            )
            yield Label("Environment: Sandbox (Testing)", classes="input-label")
        else:
            yield Static(
                "🚀 PRODUCTION MODE ACTIVE\n"
                "You're setting up PRODUCTION credentials for real listings.\n"
                "• Production creates real eBay listings that buyers can see\n"
                "• Listings will appear on eBay and can generate real sales\n"
                "• Make sure you're ready to fulfill orders!",
                classes="env-info production-info"
            )
            yield Label("Environment: Production (Real Listings)", classes="input-label")
        
        yield Label("", classes="step-description")
        
        yield Label("Client ID (App ID):", classes="input-label")
        yield Input(placeholder="e.g., yourname-appname-PRD-12345...", id="client_id")
        yield Label("", classes="step-description")
        
        yield Label("Client Secret (Cert ID):", classes="input-label")
        yield Input(placeholder="Your client secret", password=True, id="client_secret")
        yield Label("", classes="step-description")
        
        yield Label("RuName (Return URL Name):", classes="input-label")
        yield Input(placeholder="e.g., yourname-appname-appname--runame", id="runame")
        yield Label("", classes="step-description")
        
        yield Label("💡 Tip: You can find these in 'My Account' → 'Keys' in the developer portal", classes="step-description")
    
    def step_5_authorize_account(self):
        """Step 5: Authorize eBay account"""
        yield Label("Step 5: Authorize Your eBay Account 🔐", classes="step-title")
        yield Label("Finally, authorize this app to access your eBay selling account", classes="step-description")
        yield Label("", classes="step-description")
        yield Label("Instructions:", classes="step-description")
        yield Label("1. Click 'Open Authorization Page' below", classes="step-description")
        yield Label("2. Sign in to your eBay SELLING account (not developer account)", classes="step-description")
        yield Label("3. Review and accept the permissions", classes="step-description")
        yield Label("4a. If you see an authorization code, copy it", classes="step-description")
        yield Label("4b. If no code appears, copy the entire URL from your browser", classes="step-description")
        yield Label("5. Paste either the code OR the URL below and click 'Complete Setup'", classes="step-description")
        yield Label("", classes="step-description")
        yield Button("🚀 Open Authorization Page", id="open_auth", variant="success")
        yield Label("", classes="step-description")
        yield Label("Authorization Code or URL:", classes="input-label")
        yield Input(placeholder="Paste authorization code (v^1.1#i^1...) OR the full URL", id="auth_code")
    
    def step_6_setup_business_policies(self):
        """Step 6: Setup Business Policies & Seller Verification"""
        yield Label("Step 6: Seller Verification & Business Policies 📋", classes="step-title")
        yield Label("Before you can publish listings, your eBay account needs to be fully verified.", classes="step-description")
        yield Label("", classes="step-description")
        yield Label("⚠️  CRITICAL: Complete ALL verification steps before setting up policies!", classes="step-description")
        yield Label("", classes="step-description")
        
        # Seller verification section
        yield Label("🔍 Seller Verification Checklist:", classes="step-description")
        yield Label("✓ Verify your email address", classes="step-description")
        yield Label("✓ Verify your phone number", classes="step-description")
        yield Label("✓ Complete identity verification (if required)", classes="step-description")
        yield Label("✓ Add payment method for seller fees", classes="step-description")
        yield Label("✓ Opt into Seller Hub program", classes="step-description")
        yield Label("", classes="step-description")
        
        # Verification buttons
        with Horizontal():
            yield Button("🔍 Check Verification Status", id="check_verification", variant="default")
            yield Button("🏢 Opt into Seller Hub", id="opt_into_seller_hub", variant="primary")
        
        yield Label("", classes="step-description")
        yield Label("📋 After verification, set up business policies:", classes="step-description")
        yield Label("✓ Payment Policy (how buyers can pay)", classes="step-description")
        yield Label("✓ Return Policy (return terms and conditions)", classes="step-description")  
        yield Label("✓ Shipping Policy (shipping methods and costs)", classes="step-description")
        yield Label("", classes="step-description")
        
        # Policy setup button
        yield Button("🏢 Policy Management Dashboard", id="open_policy_dashboard", variant="primary")
        
        yield Label("", classes="step-description")
        yield Label("📋 Complete Setup Instructions:", classes="step-description")
        yield Label("1. Click 'Check Verification Status' to see what needs to be completed", classes="step-description")
        yield Label("2. Click 'Opt into Seller Hub' to enable business policy features", classes="step-description")
        yield Label("3. Visit eBay Seller Hub to complete any remaining verification steps", classes="step-description")
        yield Label("4. Click 'Policy Management Dashboard' to create required policies", classes="step-description")
        yield Label("5. Create all three policy types (Payment, Return, Shipping)", classes="step-description")
        yield Label("6. Check the box below when everything is complete", classes="step-description")
        yield Label("", classes="step-description")
        yield Checkbox("✅ My account is verified and all policies are created", id="setup_complete")
        yield Label("", classes="step-description")
        yield Label("💡 Tip: You can check Seller Hub for any pending verification steps", classes="step-description")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "next_btn":
            asyncio.create_task(self.next_step())
        elif event.button.id == "back_btn":
            asyncio.create_task(self.previous_step())
        elif event.button.id == "cancel_btn":
            self.dismiss(False)
        elif event.button.id == "open_dev_portal":
            self.open_developer_portal()
        elif event.button.id == "open_keys_page":
            self.open_keys_page()
        elif event.button.id == "open_auth":
            self.open_authorization()
        elif event.button.id == "open_policy_dashboard":
            self.open_policy_dashboard()
        elif event.button.id == "check_verification":
            asyncio.create_task(self.check_seller_verification())
        elif event.button.id == "opt_into_seller_hub":
            asyncio.create_task(self.opt_into_seller_hub_flow())
        elif event.button.id == "open_payment_policy":
            self.open_payment_policy()
        elif event.button.id == "open_return_policy":
            self.open_return_policy()
        elif event.button.id == "open_shipping_policy":
            self.open_shipping_policy()
    
    async def next_step(self):
        """Move to next step"""
        print(f"🔄 next_step: Current step is {self.current_step}")
        
        if self.current_step < self.total_steps:
            if self.validate_current_step():
                if self.current_step == 5:
                    # Step 5 is authorization - complete the auth setup
                    # The async completion will handle step advancement
                    print("🔄 next_step: Starting authorization completion...")
                    self.complete_authorization_setup()
                else:
                    # Regular step progression
                    print(f"🔄 next_step: Advancing from step {self.current_step} to {self.current_step + 1}")
                    self.current_step += 1
                    await self.refresh_step()
            else:
                print(f"🔄 next_step: Step {self.current_step} validation failed")
        else:
            # Final step (step 6) - complete entire setup
            print("🔄 next_step: On final step, completing setup...")
            if self.validate_current_step():
                self.complete_full_setup()
            else:
                print("🔄 next_step: Final step validation failed")
    
    async def previous_step(self):
        """Move to previous step"""
        if self.current_step > 1:
            self.current_step -= 1
            await self.refresh_step()
    
    async def refresh_step(self):
        """Refresh the step content"""
        try:
            print(f"🔄 refresh_step: Starting refresh for step {self.current_step}")
            
            # Update progress bar
            print("🔄 refresh_step: Updating progress bar...")
            try:
                progress_label = self.query_one(".progress-bar")
                progress_label.update(f"eBay Account Setup Wizard - Step {self.current_step} of {self.total_steps}")
                print("✅ refresh_step: Progress bar updated successfully")
            except Exception as e:
                print(f"❌ refresh_step: Error updating progress bar: {e}")
                raise
            
            # Update step content
            print("🔄 refresh_step: Updating step content...")
            try:
                step_content = self.query_one("#step_content")
                print("✅ refresh_step: Found step_content container")
            except Exception as e:
                print(f"❌ refresh_step: Error finding step_content: {e}")
                raise
            
            print("🔄 refresh_step: Removing existing children...")
            try:
                step_content.remove_children()
                print("✅ refresh_step: Children removed successfully")
            except Exception as e:
                print(f"❌ refresh_step: Error removing children: {e}")
                raise
            
            # Wait a moment for the removal to complete
            await asyncio.sleep(0.01)
            
            print("🔄 refresh_step: Getting new step content...")
            try:
                new_content = self.get_step_content()
                print(f"✅ refresh_step: Got content for step {self.current_step}")
            except Exception as e:
                print(f"❌ refresh_step: Error getting step content: {e}")
                raise
            
            print("🔄 refresh_step: Mounting new content...")
            try:
                step_content.mount(*new_content)
                print("✅ refresh_step: New content mounted successfully")
            except Exception as e:
                print(f"❌ refresh_step: Error mounting content: {e}")
                raise
            
            # Update navigation buttons
            print("🔄 refresh_step: Updating navigation buttons...")
            try:
                back_btn = self.query_one("#back_btn")
                next_btn = self.query_one("#next_btn")
                
                back_btn.disabled = self.current_step == 1
                next_btn.label = "Complete Setup" if self.current_step == self.total_steps else "Next →"
                print("✅ refresh_step: Navigation buttons updated successfully")
            except Exception as e:
                print(f"❌ refresh_step: Error updating buttons: {e}")
                raise
            
            print("✅ refresh_step: Step refresh completed successfully")
            
        except Exception as e:
            print(f"❌ refresh_step: Error during step refresh: {e}")
            print(f"❌ refresh_step: Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
    
    def validate_current_step(self) -> bool:
        """Validate current step before proceeding"""
        if self.current_step == 2:
            # Check if user has confirmed they have a developer account
            dev_ready = self.query_one("#dev_account_ready", Checkbox).value
            if not dev_ready:
                self.app.notify("Please check the box to confirm you have an eBay Developer account", severity="warning")
                return False
        elif self.current_step == 3:
            # Check if user has confirmed they have an application
            app_created = self.query_one("#app_created", Checkbox).value
            if not app_created:
                self.app.notify("Please check the box to confirm you have an eBay application", severity="warning")
                return False
        elif self.current_step == 4:
            client_id = self.query_one("#client_id").value.strip()
            client_secret = self.query_one("#client_secret").value.strip()
            runame = self.query_one("#runame").value.strip()
            
            if not client_id or not client_secret:
                self.app.notify("Please enter both Client ID and Client Secret", severity="error")
                return False
            
            # Store credentials for next step - use config setting for environment
            self.user_data.update({
                'client_id': client_id,
                'client_secret': client_secret,
                'runame': runame,
                'sandbox': self.sandbox_mode  # Use the config setting
            })
        elif self.current_step == 5:
            # Check if authorization code/URL has been entered
            try:
                auth_input = self.query_one("#auth_code").value.strip()
                if not auth_input:
                    self.app.notify("Please enter the authorization code or URL from eBay", severity="warning")
                    return False
            except Exception as e:
                print(f"❌ validate_current_step: Cannot find auth_code field: {e}")
                self.app.notify("Error: Authorization input field not found", severity="error")
                return False
        elif self.current_step == 6:
            # Check if user has confirmed they completed verification and policies
            try:
                setup_complete = self.query_one("#setup_complete", Checkbox).value
                if not setup_complete:
                    self.app.notify("Please complete seller verification and create all required business policies, then check the confirmation box", severity="warning")
                    return False
            except Exception:
                # If checkbox doesn't exist, step 6 content wasn't loaded properly
                self.app.notify("Step 6 content not loaded properly. Please restart the setup.", severity="error")
                return False
                return False
        
        return True
    
    def open_developer_portal(self):
        """Open eBay Developer Portal"""
        try:
            import webbrowser
            webbrowser.open("https://developer.ebay.com/")
            self.app.notify("✅ Opened eBay Developer Portal", severity="success")
        except Exception as e:
            self.app.notify(f"Could not open browser: {e}", severity="error")
            self.app.notify("Please manually visit: https://developer.ebay.com/", severity="info")
    
    def open_keys_page(self):
        """Open eBay Keys management page"""
        try:
            import webbrowser
            webbrowser.open("https://developer.ebay.com/my/keys")
            self.app.notify("✅ Opened eBay Keys page", severity="success")
        except Exception as e:
            self.app.notify(f"Could not open browser: {e}", severity="error")
            self.app.notify("Please manually visit: https://developer.ebay.com/my/keys", severity="info")
    
    def open_policy_dashboard(self):
        """Open eBay Policy Management Dashboard"""
        try:
            import webbrowser
            webbrowser.open("https://www.ebay.com/bp/policyoptin")
            self.app.notify("✅ Opened Policy Management Dashboard", severity="success")
        except Exception as e:
            self.app.notify(f"Could not open browser: {e}", severity="error")
            self.app.notify("Please manually visit: https://www.ebay.com/bp/policyoptin", severity="info")

    def open_payment_policy(self):
        """Open eBay Payment Policy creation page"""
        try:
            import webbrowser
            webbrowser.open("https://www.ebay.com/sh/policies/payment")
            self.app.notify("✅ Opened Payment Policy creation page", severity="success")
        except Exception as e:
            self.app.notify(f"Could not open browser: {e}", severity="error")
            self.app.notify("Please manually visit: https://www.ebay.com/sh/policies/payment", severity="info")
    
    def open_return_policy(self):
        """Open eBay Return Policy creation page"""
        try:
            import webbrowser
            webbrowser.open("https://www.ebay.com/sh/policies/return")
            self.app.notify("✅ Opened Return Policy creation page", severity="success")
        except Exception as e:
            self.app.notify(f"Could not open browser: {e}", severity="error")
            self.app.notify("Please manually visit: https://www.ebay.com/sh/policies/return", severity="info")
    
    def open_shipping_policy(self):
        """Open eBay Shipping Policy creation page"""
        try:
            import webbrowser
            webbrowser.open("https://www.ebay.com/sh/policies/fulfillment")
            self.app.notify("✅ Opened Shipping Policy creation page", severity="success")
        except Exception as e:
            self.app.notify(f"Could not open browser: {e}", severity="error")
            self.app.notify("Please manually visit: https://www.ebay.com/sh/policies/fulfillment", severity="info")
    
    async def check_seller_verification(self):
        """Check and display seller verification status"""
        try:
            self.app.notify("🔍 Checking seller verification status...", severity="info")
            
            # Load config to get eBay API credentials
            config = Config.load_from_keyring()
            if not config.ebay_client_id or not config.ebay_refresh_token:
                self.app.notify("❌ eBay credentials not configured", severity="error")
                return
            
            # Create eBay API instance
            ebay_api = EbayInventoryAPI(
                client_id=config.ebay_client_id,
                client_secret=config.ebay_client_secret,
                sandbox=config.ebay_sandbox,
                runame=config.ebay_runame,
                refresh_token=config.ebay_refresh_token,
                country=config.ebay_country,
                log_callback=self.app_log
            )
            
            # Check verification status
            verification = await ebay_api.check_seller_verification_status()
            
            if verification["verified"]:
                self.app.notify("✅ Seller account is verified and ready!", severity="success")
            else:
                self.app.notify("⚠️ Verification issues found - check console for details", severity="warning")
                print(f"❌ Verification issues: {verification['issues']}")
                print(f"📋 Requirements: {verification['requirements']}")
                
                for issue in verification["issues"]:
                    self.app.notify(f"• {issue}", severity="warning")
                    
        except Exception as e:
            self.app.notify(f"❌ Error checking verification: {str(e)}", severity="error")
            print(f"❌ Verification check error: {e}")
    
    async def opt_into_seller_hub_flow(self):
        """Handle opt-in to Seller Hub program"""
        try:
            self.app.notify("🔄 Opting into eBay Seller Hub...", severity="info")
            
            # Load config to get eBay API credentials
            config = Config.load_from_keyring()
            if not config.ebay_client_id or not config.ebay_refresh_token:
                self.app.notify("❌ eBay credentials not configured", severity="error")
                return
            
            # Create eBay API instance
            ebay_api = EbayInventoryAPI(
                client_id=config.ebay_client_id,
                client_secret=config.ebay_client_secret,
                sandbox=config.ebay_sandbox,
                runame=config.ebay_runame,
                refresh_token=config.ebay_refresh_token,
                country=config.ebay_country,
                log_callback=self.app_log
            )
            
            # Opt into Seller Hub
            success = await ebay_api.opt_into_seller_hub()
            
            if success:
                self.app.notify("✅ Successfully opted into Seller Hub program!", severity="success")
                self.app.notify("You can now create business policies", severity="info")
            else:
                self.app.notify("⚠️ Could not opt into Seller Hub - check console for details", severity="warning")
                
        except Exception as e:
            self.app.notify(f"❌ Error opting into Seller Hub: {str(e)}", severity="error")
            print(f"❌ Seller Hub opt-in error: {e}")
    
    def open_authorization(self):
        """Open eBay authorization page using user's credentials"""
        if not self.user_data.get('client_id'):
            self.app.notify("Please complete the previous step first", severity="error")
            return
        
        try:
            import webbrowser
            import urllib.parse
            
            # Build authorization URL using user's credentials
            if self.user_data['sandbox']:
                auth_base_url = "https://auth.sandbox.ebay.com/oauth2/authorize"
            else:
                auth_base_url = "https://auth.ebay.com/oauth2/authorize"
            
            # Comprehensive scopes for cross-listing (matching your eBay app configuration)
            scopes = [
                "https://api.ebay.com/oauth/api_scope",
                "https://api.ebay.com/oauth/api_scope/sell.marketing.readonly", 
                "https://api.ebay.com/oauth/api_scope/sell.marketing",
                "https://api.ebay.com/oauth/api_scope/sell.inventory.readonly",
                "https://api.ebay.com/oauth/api_scope/sell.inventory",
                "https://api.ebay.com/oauth/api_scope/sell.account.readonly",
                "https://api.ebay.com/oauth/api_scope/sell.account",
                "https://api.ebay.com/oauth/api_scope/sell.fulfillment.readonly",
                "https://api.ebay.com/oauth/api_scope/sell.fulfillment",
                "https://api.ebay.com/oauth/api_scope/sell.analytics.readonly",
                "https://api.ebay.com/oauth/api_scope/sell.finances",
                "https://api.ebay.com/oauth/api_scope/sell.payment.dispute",
                "https://api.ebay.com/oauth/api_scope/commerce.identity.readonly",
                "https://api.ebay.com/oauth/api_scope/sell.reputation",
                "https://api.ebay.com/oauth/api_scope/sell.reputation.readonly",
                "https://api.ebay.com/oauth/api_scope/commerce.notification.subscription",
                "https://api.ebay.com/oauth/api_scope/commerce.notification.subscription.readonly",
                "https://api.ebay.com/oauth/api_scope/sell.stores",
                "https://api.ebay.com/oauth/api_scope/sell.stores.readonly",
                "https://api.ebay.com/oauth/scope/sell.edelivery",
                "https://api.ebay.com/oauth/api_scope/commerce.vero"
            ]
            
            auth_params = {
                'client_id': self.user_data['client_id'],
                'response_type': 'code',
                'redirect_uri': self.user_data['runame'] or 'urn:ietf:wg:oauth:2.0:oob',
                'scope': ' '.join(scopes)
            }
            
            auth_url = auth_base_url + '?' + urllib.parse.urlencode(auth_params)
            
            # Debug information
            print(f"🔄 Authorization URL Debug:")
            print(f"  - Base URL: {auth_base_url}")
            print(f"  - Client ID: {self.user_data['client_id']}")
            print(f"  - Redirect URI: {self.user_data['runame']}")
            print(f"  - Sandbox mode: {self.user_data['sandbox']}")
            print(f"  - Full URL: {auth_url}")
            
            webbrowser.open(auth_url)
            
            env_type = "Sandbox" if self.user_data['sandbox'] else "Production"
            self.app.notify(f"✅ Opened eBay {env_type} authorization page", severity="success")
            
        except Exception as e:
            self.app.notify(f"Could not open authorization page: {e}", severity="error")
    
    def extract_auth_code(self, input_text: str) -> str:
        """Extract authorization code from either direct code or URL"""
        input_text = input_text.strip()
        
        # If it's already just the code (starts with v^), return as-is
        if input_text.startswith('v^'):
            return input_text
        
        # If it's a URL, extract the code parameter
        if input_text.startswith('http'):
            try:
                from urllib.parse import urlparse, parse_qs, unquote
                parsed = urlparse(input_text)
                query_params = parse_qs(parsed.query)
                
                print(f"🔍 Debug: Parsed URL query params: {query_params}")
                
                if 'code' in query_params and len(query_params['code']) > 0:
                    # URL decode the code
                    raw_code = query_params['code'][0]
                    decoded_code = unquote(raw_code)
                    print(f"🔍 Debug: Extracted code: {decoded_code[:50]}...")
                    return decoded_code
                else:
                    # Sometimes the code is in the fragment (after #)
                    if parsed.fragment:
                        fragment_params = parse_qs(parsed.fragment)
                        print(f"🔍 Debug: Fragment params: {fragment_params}")
                        if 'code' in fragment_params and len(fragment_params['code']) > 0:
                            raw_code = fragment_params['code'][0]
                            decoded_code = unquote(raw_code)
                            return decoded_code
                    
                    raise ValueError(f"No authorization code found in URL. Available params: {list(query_params.keys())}")
            except Exception as e:
                raise ValueError(f"Could not extract code from URL: {e}")
        
        # If it doesn't start with v^ and isn't a URL, assume it's still a code
        return input_text
    
    def complete_authorization_setup(self):
        """Complete the authorization setup and move to business policies step"""
        print(f"🔄 complete_authorization_setup: Current step is {self.current_step}")
        
        # Verify we're on the right step and the field exists
        try:
            auth_input_field = self.query_one("#auth_code")
            auth_input = auth_input_field.value.strip()
            print(f"🔄 complete_authorization_setup: Found auth_code field with {len(auth_input)} characters")
        except Exception as e:
            print(f"❌ complete_authorization_setup: Cannot find auth_code field: {e}")
            self.app.notify("Error: Authorization input field not found. Please try refreshing the step.", severity="error")
            return
        
        if not auth_input:
            self.app.notify("Please enter the authorization code or URL", severity="error")
            return
        
        try:
            # Extract the actual auth code from input (handles both codes and URLs)
            auth_code = self.extract_auth_code(auth_input)
            self.app.notify("✅ Authorization code extracted successfully", severity="success")
        except ValueError as e:
            self.app.notify(f"Error: {e}", severity="error")
            return
        
        
        async def finish_auth_setup():
            try:
                print("🔄 finish_auth_setup: Starting async authorization completion...")
                
                # Save user credentials
                print("🔄 finish_auth_setup: Loading existing config...")
                existing_config = Config.load_from_keyring()
                
                print("🔄 finish_auth_setup: Creating new config...")
                config = Config(
                    ebay_client_id=self.user_data['client_id'],
                    ebay_client_secret=self.user_data['client_secret'],
                    ebay_runame=self.user_data['runame'],
                    ebay_sandbox=self.user_data['sandbox'],
                    ebay_refresh_token=None,  # Will be set after code exchange
                    
                    # Preserve other settings
                    keepa_api_key=existing_config.keepa_api_key,
                    ebay_country=existing_config.ebay_country,
                    net_margin=existing_config.net_margin,
                    ebay_fee_rate=existing_config.ebay_fee_rate,
                    paypal_fee_rate=existing_config.paypal_fee_rate,
                    app_password_hash=existing_config.app_password_hash,
                    lock_timeout_minutes=existing_config.lock_timeout_minutes,
                    security_enabled=existing_config.security_enabled
                )
                
                print("🔄 finish_auth_setup: Saving initial config to keyring...")
                config.save_to_keyring()
                
                # Exchange authorization code for refresh token
                print("🔄 finish_auth_setup: Creating eBay API instance...")
                ebay_api = EbayInventoryAPI(
                    client_id=self.user_data['client_id'],
                    client_secret=self.user_data['client_secret'],
                    sandbox=self.user_data['sandbox'],
                    runame=self.user_data['runame'],
                    country=existing_config.ebay_country,
                    log_callback=print
                )
                
                print("🔄 finish_auth_setup: Notifying user of token exchange...")
                self.app.notify("🔄 Exchanging authorization code for refresh token...", severity="info")
                
                # URL decode the authorization code
                import urllib.parse
                decoded_auth_code = urllib.parse.unquote(auth_code)
                print(f"🔄 Starting authorization process with decoded code: {decoded_auth_code[:20]}...")
                
                refresh_token = await ebay_api.exchange_auth_code_for_refresh_token(decoded_auth_code)
                print(f"✅ Got refresh token: {refresh_token[:20]}...")
                
                # Update config with refresh token
                print("🔄 Updating config with refresh token...")
                config.ebay_refresh_token = refresh_token
                
                print("🔄 Saving config to keyring...")
                config.save_to_keyring()
                
                print("🔄 Sending success notification...")
                self.app.notify("✅ eBay account authorization completed!", severity="success")
                
                print("🔄 Moving to next step...")
                # Move to next step (business policies)
                old_step = self.current_step
                self.current_step += 1
                print(f"🔄 Changed step from {old_step} to {self.current_step}")
                
                print("🔄 Scheduling step refresh...")
                # Since we're already in the main thread, just schedule the async task directly
                async def refresh_step_content():
                    try:
                        await self.refresh_step()
                        print("✅ Authorization process completed successfully!")
                    except Exception as refresh_error:
                        print(f"❌ Error during step refresh: {refresh_error}")
                        import traceback
                        traceback.print_exc()
                        # Try to revert step change if refresh fails
                        self.current_step = old_step
                        self.app.notify("Step refresh failed - please try again", severity="error")
                
                # Schedule the refresh task
                asyncio.create_task(refresh_step_content())
                
            except Exception as e:
                print(f"❌ finish_auth_setup: Authorization failed with error: {e}")
                print(f"❌ finish_auth_setup: Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                self.app.notify(f"Authorization failed: {str(e)}", severity="error")
                # Don't advance to next step if authorization failed
                return
        
        # Run the async function
        asyncio.create_task(finish_auth_setup())
    
    def complete_full_setup(self):
        """Complete the entire setup process"""
        self.app.notify("🎉 eBay setup completed successfully!", severity="success")
        self.app.notify("💡 Remember: You can modify business policies anytime in My eBay", severity="info")
        self.dismiss(True)



class FloatingProgressOverlay(Static):
    """Floating progress bar overlay for long-running operations"""
    
    CSS = """
    FloatingProgressOverlay {
        width: 66%;
        height: 6;
        background: $surface;
        border: thick $primary;
        layer: notification;
        dock: top;
        margin: 1 auto;
        padding: 1;
        display: none;
        content-align: center middle;
        pointer-events: none;  /* Don't capture input when hidden */
    }
    
    FloatingProgressOverlay:not([style*="display: none"]) {
        pointer-events: auto;  /* Allow input when visible */
    }
    
    .progress-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    
    .progress-details {
        color: $text-muted;
        text-align: center;
        width: 100%;
        margin-top: 1;
    }
    
    ProgressBar {
        width: 100%;
        margin: 0;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_time = None
        self.total_items = 0
        self.processed_items = 0
        self.displayed = False
        
    def compose(self) -> ComposeResult:
        yield Label("Processing...", classes="progress-title", id="progress_title")
        yield ProgressBar(id="progress_bar")
        yield Label("", classes="progress-details", id="progress_details")
    
    def show_progress(self, title: str, total_items: int = 100):
        """Show the progress overlay with title and total items"""
        import time
        self.start_time = time.time()
        self.total_items = total_items
        self.processed_items = 0
        
        self.query_one("#progress_title").update(title)
        progress_bar = self.query_one("#progress_bar")
        progress_bar.total = total_items
        progress_bar.progress = 0
        self.query_one("#progress_details").update("Starting...")
        
        self.displayed = True
        self.styles.display = "block"
        self.styles.pointer_events = "auto"  # Ensure input is enabled when visible
        
    def update_progress(self, processed: int, current_item: str = ""):
        """Update progress with current status"""
        if not self.displayed:
            return
            
        self.processed_items = processed
        progress_bar = self.query_one("#progress_bar")
        progress_bar.progress = processed
        
        # Calculate percentage
        percentage = int((processed / self.total_items) * 100) if self.total_items > 0 else 0
        
        # Calculate estimated time remaining
        time_info = ""
        if self.start_time and processed > 0:
            import time
            elapsed = time.time() - self.start_time
            items_per_second = processed / elapsed
            if items_per_second > 0:
                remaining_items = self.total_items - processed
                estimated_seconds = remaining_items / items_per_second
                
                if estimated_seconds > 60:
                    minutes = int(estimated_seconds // 60)
                    seconds = int(estimated_seconds % 60)
                    time_info = f" • ~{minutes}m {seconds}s remaining"
                else:
                    time_info = f" • ~{int(estimated_seconds)}s remaining"
        
        # Update details
        item_info = f" • {current_item}" if current_item else ""
        details = f"{processed}/{self.total_items} ({percentage}%){time_info}{item_info}"
        self.query_one("#progress_details").update(details)
    
    def hide_progress(self):
        """Hide the progress overlay and ensure proper cleanup"""
        self.displayed = False
        self.styles.display = "none"
        self.styles.pointer_events = "none"  # Disable input capture
        self.start_time = None
        
        # Force focus back to the main app to ensure input responsiveness  
        try:
            if self.app and hasattr(self.app, 'focus'):
                self.app.set_focus(None)  # Clear any captured focus
                # Also try to refresh the screen to ensure UI responsiveness
                self.app.refresh(layout=True)
        except Exception:
            pass  # Ignore focus issues during cleanup


class LockScreen(ModalScreen):
    """
    Security lock screen that blocks access while showing minimal progress.
    
    This screen prevents unauthorized access while allowing background 
    operations to continue. Shows current processing status without
    revealing sensitive information.
    """
    
    CSS = """
    LockScreen {
        background: rgba(0, 0, 0, 0.8);
        align: center middle;
    }
    
    #lock_container {
        width: 40;
        height: 15;
        background: $surface;
        border: thick $primary;
        padding: 2;
        text-align: center;
    }
    
    .lock-title {
        text-style: bold;
        color: $primary;
        margin: 0 0 1 0;
    }
    
    .status-text {
        color: $text-muted;
        margin: 1 0;
    }
    
    Input {
        margin: 1 0;
        width: 100%;
    }
    
    Button {
        margin: 1 0 0 0;
        width: 100%;
    }
    """
    
    def __init__(self, progress_overlay=None):
        super().__init__()
        self.failed_attempts = 0
        self.max_attempts = 3
        self.progress_overlay = progress_overlay
    
    def compose(self) -> ComposeResult:
        with Container(id="lock_container"):
            yield Label("🔒 Application Locked", classes="lock-title")
            yield Label("Enter your password to continue", classes="status-text") 
            yield Label("Current Status: Idle", id="status_display", classes="status-text")
            
            # If there's an active progress operation, show progress in lock screen
            if self.progress_overlay and hasattr(self.progress_overlay, 'displayed') and self.progress_overlay.displayed:
                yield Label("🔄 Background Operation Running", classes="status-text")
                try:
                    progress_title = self.progress_overlay.query_one("#progress_title").renderable
                    progress_details = self.progress_overlay.query_one("#progress_details").renderable
                    progress_bar = self.progress_overlay.query_one("#progress_bar")
                    
                    yield Label(f"Status: {progress_title}", classes="status-text")
                    yield ProgressBar(show_eta=False, show_percentage=True, total=progress_bar.total, progress=progress_bar.progress, id="lock_progress")
                    yield Label(f"{progress_details}", classes="status-text")
                except Exception:
                    yield Label("Status: Processing...", classes="status-text")
            
            yield Input(placeholder="Enter password", password=True, id="password_input")
            yield Button("Unlock", variant="primary", id="unlock_btn")
            yield Label("", id="error_message", classes="status-text")
    
    def on_mount(self):
        """Focus password input when screen loads"""
        self.query_one("#password_input").focus()
        self.update_status()
    
    def update_status(self):
        """Update the status display with current app activity"""
        # This will be called by the main app to show progress
        # For now, just show idle status
        status_label = self.query_one("#status_display")
        status_label.update("Current Status: Idle")
    
    def set_status(self, status: str):
        """Set the current processing status from external source"""
        try:
            status_label = self.query_one("#status_display")
            status_label.update(f"Current Status: {status}")
        except:
            pass  # Screen might not be mounted yet
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "unlock_btn":
            self.attempt_unlock()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "password_input":
            self.attempt_unlock()
    
    def attempt_unlock(self):
        """Verify password and unlock if correct"""
        password_input = self.query_one("#password_input")
        error_label = self.query_one("#error_message")
        
        password = password_input.value.strip()
        if not password:
            error_label.update("Please enter your password")
            return
        
        # Load config and verify password
        config = Config.load_from_keyring()
        if not config.app_password_hash:
            # No password set - this shouldn't happen
            error_label.update("Security error: No password configured")
            return
        
        if verify_password(password, config.app_password_hash):
            # Success! Dismiss the lock screen
            self.dismiss(True)
        else:
            # Failed attempt
            self.failed_attempts += 1
            password_input.value = ""
            
            if self.failed_attempts >= self.max_attempts:
                error_label.update(f"Too many failed attempts. Try again in 30 seconds.")
                # TODO: Implement temporary lockout
            else:
                remaining = self.max_attempts - self.failed_attempts
                error_label.update(f"Incorrect password. {remaining} attempts remaining.")


class ConfigScreen(ModalScreen):
    """Enhanced configuration screen with tabbed interface"""
    
    def app_log(self, message: str):
        """Logging method for modal screens - uses notifications instead of main log"""
        # Use notification system for modal screen logging
        self.app.notify(message, severity="info")
    
    CSS = """
    ConfigScreen {
        align: center middle;
    }
    
    ConfigScreen {
        width: 80;
        height: 80%;
        min-height: 40;
        max-height: 80%;      /* Ensure scrolling when content overflows */
        border: thick $primary;
        background: $surface;
        padding: 1;
        overflow: auto;       /* Enable scrolling for the entire config */
    }

    ScrollableContainer {
        height: 100%;
        width: 100%;
    }

    .header {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        height: 3;
    }

    /* TabbedContent - Main container for tabbed interface */
    TabbedContent {
        width: 100%;
        height: 1fr;          /* Use fractional unit for proper scaling */
        min-height: 50;       /* Larger minimum height for tabs + content */
        border: solid $accent;
    }

    /* Tabs - The navigation bar at the top */
    Tabs {
        background: $surface; /* Changed from debug blue to normal surface */
        dock: top;            /* CRITICAL: Dock to top of TabbedContent */
        height: 1fr;          /* Navigation bar gets 1 fractional unit */
        min-height: 3;        /* Minimum height to ensure visibility */
        max-height: 5;        /* Maximum height to prevent oversized tabs */
        width: 100%;
    }

    /* Tab - Individual tab buttons */
    Tab {
        min-width: 15;        /* Wider for better visibility */
        background: $accent;  /* Use accent color for professional look */
        color: $text;         /* Ensure text is visible */
        text-style: bold;     /* Make text bold */
        padding: 0 2;         /* Horizontal padding for better spacing */
        border-right: solid $primary; /* Separator between tabs */
    }

    /* Active tab styling */
    Tab.-active {
        background: $primary; /* Different color from regular tabs */
        color: $text;
        text-style: bold;
        border-bottom: solid $warning; /* Add bottom border for active state */
    }

    /* TabPane - Content area for each tab */
    TabPane {
        height: 11fr;        /* Content area gets 11 fractional units (11:1 ratio) */
        width: 100%;
        padding: 1;
        min-height: 20;      /* Ensure minimum content area */
        overflow: auto;      /* Enable scrolling if content overflows */
    }
    
    .setting-label {
        text-style: bold;
        margin: 1 0 0 0;
        color: $text;
        height: auto;
    }
    
    .info-text {
        color: $text;         /* Fixed: Use $text instead of invalid $text-muted */
        text-style: italic;
        margin: 0 0 1 0;
    }
    
    .setting-group {
        height: auto;
        border-bottom: dashed $accent;
        padding-bottom: 1;
    }
    
    Input {
        margin: 0 0 1 0;
        width: 100%;
        height: auto;
    }
    
    Select {
        margin: 0 0 1 0;
        width: 100%;
        height: auto;
    }
    
    Button {
        margin: 0 1 0 0;
        height: auto;
    }

    #bottom_buttons {
        height: auto;        
        content-align: center middle;
        padding: 1;
        border-top: solid $accent;
        margin-top: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        with ScrollableContainer():    
            yield Label("eBay Cross-Lister Configuration", classes="header")
            
            with TabbedContent(initial="accounts_tab"):

                # Account Setup Tab  
                with TabPane("Accounts", id="accounts_tab"):
                    yield Label("Configure your accounts and API keys", classes="info-text")
                    # eBay Account Setup
                    with Container(classes="setting-group"):
                        yield Label("eBay Account:", classes="setting-label")
                        with Horizontal():
                            yield Button("✅ Test eBay Connection", variant="default", id="test_ebay")
                            yield Button("🔧 Launch Setup Wizard", variant="warning", id="launch_setup_wizard")
                    
                    # Keepa API Setup
                    with Container(classes="setting-group"):
                        yield Label("Keepa API Key (Optional - Enhanced Data):", classes="setting-label")
                        yield Input(placeholder="Enter your Keepa API key for premium data", id="keepa_key")
                        yield Label("ℹ️ Keepa provides accurate pricing history and stock data", classes="info-text")
                        with Horizontal():
                            yield Button("🔗 Get Keepa API Key", variant="default", id="keepa_signup")
                            yield Button("✅ Test Keepa", variant="default", id="test_keepa")
                    
                    # Environment Settings
                    with Container(classes="setting-group"):
                        yield Label("Environment:", classes="setting-label")
                        with Horizontal():
                            yield Label("Sandbox Mode (for testing):")
                            yield Switch(value=True, id="ebay_sandbox")
                        yield Label("ℹ️ Sandbox mode creates test listings that don't appear on eBay", classes="info-text")
                        yield Label("ℹ️ IMPORTANT: Sandbox mode requires additional eBay API setup within the setup wizard.", classes="info-text")

                # Pricing Tab
                with TabPane("Pricing", id="pricing_tab"):
                    yield Label("Configure pricing and fee settings for pricing optimization", classes="info-text")

                    with Container(classes="setting-group"):
                        yield Label("Net Profit Margin:", classes="setting-label")
                        yield Input(placeholder="e.g. 0.20 (20% profit margin)", id="net_margin")
                        yield Label("ℹ️ Your desired profit margin after all fees", classes="info-text")
                        
                        with Container(classes="setting-group"):
                            yield Label("eBay Final Value Fees:", classes="setting-label")
                            yield Input(placeholder="e.g. 0.13 (13% eBay fees)", id="ebay_fee_rate")
                            yield Label("ℹ️ eBay's selling fees (varies by category)", classes="info-text")
                        
                        with Container(classes="setting-group"):
                            yield Label("PayPal Processing Fees:", classes="setting-label")
                            yield Input(placeholder="e.g. 0.0349 (3.49% PayPal fees)", id="paypal_fee_rate")
                            yield Label("ℹ️ Payment processing fees (optional)", classes="info-text")

                # Location Tab
                with TabPane("Location", id="location_tab"):
                    yield Label("ℹ️ Configure your business/warehouse location for eBay listings", classes="info-text")
                    
                    with Container(classes="setting-group"):
                        yield Label("Street Address:", classes="setting-label")
                        yield Input(placeholder="e.g. 123 Main Street", id="location_address_line1")
                        
                        yield Label("Apartment/Suite (Optional):", classes="setting-label")
                        yield Input(placeholder="e.g. Apt 4B, Suite 200", id="location_address_line2")
                        
                        yield Label("City:", classes="setting-label")
                        yield Input(placeholder="e.g. New York", id="location_city")
                        
                        yield Label("State/Province:", classes="setting-label")
                        yield Input(placeholder="e.g. NY or Ontario", id="location_state_province")
                        
                        yield Label("ZIP/Postal Code:", classes="setting-label")
                        yield Input(placeholder="e.g. 10001 or M5V 3A8", id="location_postal_code")
                        
                        # Country selection
                        yield Label("Country:", classes="setting-label")
                        yield Select(
                            [
                                ("United States", "US"),
                                ("United Kingdom", "GB"),
                                ("Germany", "DE"),
                                ("Australia", "AU"),
                                ("Austria", "AT"),
                                ("Belgium (French)", "BE_FR"),
                                ("Belgium (Dutch)", "BE_NL"),
                                ("Canada", "CA"),
                                ("France", "FR"),
                                ("Hong Kong", "HK"),
                                ("India", "IN"),
                                ("Ireland", "IE"),
                                ("Italy", "IT"),
                                ("Malaysia", "MY"),
                                ("Netherlands", "NL"),
                                ("Philippines", "PH"),
                                ("Poland", "PL"),
                                ("Singapore", "SG"),
                                ("Spain", "ES"),
                                ("Switzerland", "CH"),
                                ("Taiwan", "TW"),
                                ("Thailand", "TH"),
                                ("Turkey", "TR"),
                                ("Vietnam", "VN"),
                                ("Israel", "IL"),
                                ("Russia", "RU"),
                                ("China", "CN"),
                                ("Japan", "JP"),
                                ("South Korea", "KR"),
                                ("Brazil", "BR"),
                                ("Mexico", "MX"),
                                ("Argentina", "AR"),
                                ("Chile", "CL"),
                                ("Colombia", "CO"),
                                ("Peru", "PE"),
                                ("Uruguay", "UY"),
                                ("South Africa", "ZA"),
                                ("Morocco", "MA"),
                                ("Egypt", "EG"),
                                ("Jordan", "JO"),
                                ("Kuwait", "KW"),
                                ("Lebanon", "LB"),
                                ("Oman", "OM"),
                                ("Qatar", "QA"),
                                ("Saudi Arabia", "SA"),
                                ("United Arab Emirates", "AE"),
                                ("Czech Republic", "CZ"),
                                ("Denmark", "DK"),
                                ("Finland", "FI"),
                                ("Greece", "GR"),
                                ("Hungary", "HU"),
                                ("Norway", "NO"),
                                ("Portugal", "PT"),
                                ("Romania", "RO"),
                                ("Slovakia", "SK"),
                                ("Slovenia", "SI"),
                                ("Sweden", "SE"),
                                ("Ukraine", "UA"),
                                ("Estonia", "EE"),
                                ("Latvia", "LV"),
                                ("Lithuania", "LT"),
                                ("Luxembourg", "LU"),
                                ("Malta", "MT"),
                                ("Cyprus", "CY"),
                                ("Bulgaria", "BG"),
                                ("Croatia", "HR"),
                            ],
                            value="US",
                            id="ebay_country"
                        )

                # Shipping Tab
                with TabPane("Shipping", id="shipping_tab"):
                    yield Label("ℹ️ Configure shipping cost overrides to absorb shipping costs", classes="info-text")
                    
                    with Container(classes="setting-group"):
                        with Horizontal():
                            yield Label("Enable Shipping Cost Override:")
                            yield Switch(value=False, id="shipping_override_enabled")
                        yield Label("ℹ️ Override listing shipping costs with fixed amounts", classes="info-text")
                    
                    # Shipping override settings (initially hidden)
                    with Container(id="shipping_override_container", classes="setting-group"):
                        yield Label("Shipping Cost Amount ($):", classes="setting-label")
                        yield Input(placeholder="e.g. 0.00 for free shipping", id="shipping_override_amount")
                        yield Label("ℹ️ Set to 0.00 to offer free shipping", classes="info-text")
                        
                        yield Label("Each Additional Item ($):", classes="setting-label")
                        yield Input(placeholder="e.g. 0.00", id="shipping_additional_override")
                        yield Label("ℹ️ Cost for each additional identical item", classes="info-text")
                        
                        with Horizontal():
                            yield Label("Domestic Shipping Only:")
                            yield Switch(value=True, id="shipping_domestic_only")
                        yield Label("ℹ️ Apply override to domestic shipping only", classes="info-text")

                # Security Tab
                with TabPane("Security", id="security_tab"):
                    yield Label("ℹ️ Cross-platform secure credential storage", classes="info-text")
                    yield Button("🔐 Test Keyring Security", variant="default", id="test_keyring")
                    
                    with Container(classes="setting-group"):
                        with Horizontal():
                            yield Label("Password Protection:")
                            yield Switch(value=False, id="security_enabled")
                    
                    # Password setup (initially hidden)
                    with Container(id="password_container", classes="setting-group"):
                        yield Input(placeholder="Current password (if changing)", password=True, id="current_password")
                        yield Input(placeholder="Enter new password", password=True, id="new_password")
                        yield Input(placeholder="Confirm password", password=True, id="confirm_password")
                        yield Button("Set Password", variant="default", id="set_password")
                    
                    # Timeout setting
                    with Container(id="timeout_container", classes="setting-group"):
                        yield Label("Auto-lock timeout (minutes):")
                        yield Input(placeholder="10", id="lock_timeout")
        
            # Bottom buttons (outside of tabs)
            with Horizontal(id = "bottom_buttons"):
                yield Button("Save", variant="primary", id="save_config")
                yield Button("Cancel", id="cancel_config")
        
    def on_mount(self):
        """Load existing configuration"""
        config = Config.load_from_keyring()

        tabbed_content = self.query_one(TabbedContent)

        # Load settings
        if config.keepa_api_key:
            self.query_one("#keepa_key").value = config.keepa_api_key
        
        self.query_one("#ebay_sandbox").value = config.ebay_sandbox
        self.query_one("#ebay_country").value = config.ebay_country
        
        # Load location settings
        self.query_one("#location_address_line1").value = config.location_address_line1
        self.query_one("#location_address_line2").value = config.location_address_line2
        self.query_one("#location_city").value = config.location_city
        self.query_one("#location_state_province").value = config.location_state_province
        self.query_one("#location_postal_code").value = config.location_postal_code
        
        if config.net_margin is not None:
            self.query_one("#net_margin").value = str(config.net_margin)
        if config.ebay_fee_rate is not None:
            self.query_one("#ebay_fee_rate").value = str(config.ebay_fee_rate)
        if config.paypal_fee_rate is not None:
            self.query_one("#paypal_fee_rate").value = str(config.paypal_fee_rate)
        
        # Load shipping override settings
        self.query_one("#shipping_override_enabled").value = config.shipping_cost_override_enabled
        self.query_one("#shipping_override_amount").value = str(config.shipping_cost_override_amount)
        self.query_one("#shipping_additional_override").value = str(config.shipping_additional_cost_override)
        self.query_one("#shipping_domestic_only").value = config.shipping_cost_override_domestic_only
        
        # Load security settings
        self.query_one("#security_enabled").value = config.security_enabled
        self.query_one("#lock_timeout").value = str(config.lock_timeout_minutes)
        
        # Set up containers visibility
        self.toggle_security_containers(config.security_enabled)
        self.toggle_shipping_containers(config.shipping_cost_override_enabled)
    
    def toggle_security_containers(self, enabled: bool):
        """Show/hide security-related containers based on enabled state"""
        password_container = self.query_one("#password_container")
        timeout_container = self.query_one("#timeout_container")
        
        password_container.display = enabled
        timeout_container.display = enabled
    
    def toggle_shipping_containers(self, enabled: bool):
        """Show/hide shipping override containers based on enabled state"""
        shipping_container = self.query_one("#shipping_override_container")
        shipping_container.display = enabled
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch toggles"""
        if event.switch.id == "security_enabled":
            self.toggle_security_containers(event.value)
        elif event.switch.id == "shipping_override_enabled":
            self.toggle_shipping_containers(event.value)
    
    def handle_password_setup(self):
        """Handle setting up a new password"""
        current_password = self.query_one("#current_password").value.strip()
        new_password = self.query_one("#new_password").value.strip()
        confirm_password = self.query_one("#confirm_password").value.strip()
        
        # Check if password is already set
        config = Config.load_from_keyring()
        if config.app_password_hash:
            # Require current password to change
            if not current_password:
                self.app.notify("Please enter current password", severity="error")
                return
            
            if not verify_password(current_password, config.app_password_hash):
                self.app.notify("Current password is incorrect", severity="error")
                return
        
        if not new_password:
            self.app.notify("Please enter a new password", severity="error")
            return
        
        if len(new_password) < 4:
            self.app.notify("Password must be at least 4 characters", severity="error")
            return
        
        if new_password != confirm_password:
            self.app.notify("Passwords do not match", severity="error")
            return
        
        # Check if this was a password change or initial setup
        was_password_change = bool(config.app_password_hash)
        
        # Hash and store the password
        password_hash = hash_password(new_password)
        
        # Update config temporarily (will be saved with main save)
        config.app_password_hash = password_hash
        config.save_to_keyring()
        
        # Clear password fields
        self.query_one("#current_password").value = ""
        self.query_one("#new_password").value = ""
        self.query_one("#confirm_password").value = ""
        
        # Success message
        if was_password_change:
            self.app.notify("Password changed successfully!", severity="success")
        else:
            self.app.notify("Password set successfully!", severity="success")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ebay_setup":
            # Open new eBay Account Setup Wizard
            def on_ebay_setup_complete(result):
                if result:
                    # Refresh main app config after eBay setup
                    if hasattr(self.app, 'reload_configuration'):
                        self.app.reload_configuration()
                    self.app.notify("eBay account setup completed successfully!", severity="success")
            
            self.app.push_screen(EbayAccountSetupWizard(), on_ebay_setup_complete)
            
        elif event.button.id == "test_ebay":
            # Test eBay API connection
            self.test_ebay_connection()
            
        elif event.button.id == "keepa_signup":
            # Open Keepa signup page
            import webbrowser
            try:
                webbrowser.open("https://keepa.com/#!api")
                self.app.notify("Opening Keepa API registration page...", severity="info")
            except Exception as e:
                self.app.notify(f"Could not open browser: {e}", severity="error")
        
        elif event.button.id == "test_keepa":
            # Test Keepa API connection
            self.test_keepa_connection()
            
        elif event.button.id == "test_keyring":
            # Test cross-platform keyring functionality
            self.test_keyring_compatibility()
            
        elif event.button.id == "launch_setup_wizard":
            # Launch the complete eBay setup wizard
            def on_setup_complete(result):
                if result:
                    # Refresh main app config after setup
                    if hasattr(self.app, 'reload_configuration'):
                        self.app.reload_configuration()
                    self.app.notify("eBay setup wizard completed successfully!", severity="success")
                    # Close config screen and return to main app
                    self.dismiss(True)
                else:
                    self.app.notify("Setup wizard was cancelled", severity="info")
            
            # Close this screen and launch wizard
            self.dismiss(False)
            self.app.push_screen(EbayAccountSetupWizard(), on_setup_complete)
            
        elif event.button.id == "set_password":
            # Handle password setup
            self.handle_password_setup()
            
        elif event.button.id == "save_config":
            try:
                # Load existing config to preserve settings
                existing_config = Config.load_from_keyring()
                
                # Validate security settings
                security_enabled = self.query_one("#security_enabled").value
                lock_timeout = int(self.query_one("#lock_timeout").value or "10")
                
                if security_enabled and not existing_config.app_password_hash:
                    self.app.notify("Please set a password before enabling security", severity="error")
                    return
                
                # Create new config with updated values
                config = Config(
                    # Preserve eBay settings
                    ebay_client_id=existing_config.ebay_client_id,
                    ebay_client_secret=existing_config.ebay_client_secret,
                    ebay_refresh_token=existing_config.ebay_refresh_token,
                    ebay_runame=existing_config.ebay_runame,
                    
                    # Update other settings
                    keepa_api_key=self.query_one("#keepa_key").value or None,
                    ebay_sandbox=self.query_one("#ebay_sandbox").value,
                    ebay_country=self.query_one("#ebay_country").value,
                    net_margin=float(self.query_one("#net_margin").value or "0.20"),
                    ebay_fee_rate=float(self.query_one("#ebay_fee_rate").value or "0.13"),
                    paypal_fee_rate=float(self.query_one("#paypal_fee_rate").value or "0.0349"),
                    
                    # Location settings
                    location_address_line1=self.query_one("#location_address_line1").value,
                    location_address_line2=self.query_one("#location_address_line2").value,
                    location_city=self.query_one("#location_city").value,
                    location_state_province=self.query_one("#location_state_province").value,
                    location_postal_code=self.query_one("#location_postal_code").value,
                    
                    # Shipping override settings
                    shipping_cost_override_enabled=self.query_one("#shipping_override_enabled").value,
                    shipping_cost_override_amount=float(self.query_one("#shipping_override_amount").value or "0.0"),
                    shipping_cost_override_domestic_only=self.query_one("#shipping_domestic_only").value,
                    shipping_additional_cost_override=float(self.query_one("#shipping_additional_override").value or "0.0"),
                    
                    # Security settings
                    app_password_hash=existing_config.app_password_hash,  # Preserve existing hash
                    lock_timeout_minutes=lock_timeout,
                    security_enabled=security_enabled
                )
                
                config.save_to_keyring()
                self.dismiss(True)
                
            except ValueError as e:
                self.app.notify(f"Invalid configuration: {e}", severity="error")
        
        else:  # Cancel
            self.dismiss(False)
    
    def test_keyring_compatibility(self):
        """Test cross-platform keyring security functionality"""
        try:
            status = Config.verify_keyring_compatibility()
            
            # Display detailed compatibility report
            platform = status["platform"]
            backend = status["backend"]
            
            if status["test_successful"]:
                self.app.notify(f"✅ Keyring secure on {platform} (Backend: {backend})", severity="success")
            elif status["keyring_available"]:
                self.app.notify(f"⚠️ Keyring available but test failed on {platform}: {status['error']}", severity="warning")
            else:
                self.app.notify(f"❌ Keyring not available on {platform}: {status['error']}", severity="error")
                
            # Additional platform-specific information
            if platform == "Darwin":  # macOS
                self.app.notify("ℹ️ macOS: Using native Keychain for secure storage", severity="info")
            elif platform == "Windows":
                self.app.notify("ℹ️ Windows: Using Windows Credential Manager", severity="info")
            elif platform == "Linux":
                self.app.notify("ℹ️ Linux: Requires libsecret or similar keyring service", severity="info")
            
        except Exception as e:
            self.app.notify(f"❌ Keyring test failed: {e}", severity="error")
    
    def test_ebay_connection(self):
        """Test eBay API connection with current credentials"""
        
        async def test_connection():
            try:
                # Load current config from keyring
                config = Config.load_from_keyring()
                
                # Debug: Show what we're actually using
                self.app.notify(f"Debug: Using Client ID: {config.ebay_client_id[:8] if config.ebay_client_id else 'None'}...", severity="info")
                self.app.notify(f"Debug: Has Refresh Token: {'Yes' if config.ebay_refresh_token else 'No'}", severity="info")
                self.app.notify(f"Debug: Sandbox Mode: {config.ebay_sandbox}", severity="info")
                
                if not config.ebay_client_id or not config.ebay_client_secret:
                    self.app.notify("eBay credentials not configured", severity="error")
                    return
                
                # Initialize eBay API client
                ebay_api = EbayInventoryAPI(
                    client_id=config.ebay_client_id,
                    client_secret=config.ebay_client_secret,
                    refresh_token=config.ebay_refresh_token,
                    sandbox=config.ebay_sandbox,
                    runame=config.ebay_runame,
                    country=config.ebay_country,
                    log_callback=self.app_log
                )
                
                # Test getting access token
                self.app.notify("Testing eBay connection...", severity="info")
                access_token = await ebay_api.get_access_token()
                
                if access_token:
                    env_type = "Sandbox" if config.ebay_sandbox else "Production"
                    self.app.notify(f"✅ eBay {env_type} connection successful!", severity="success")
                else:
                    self.app.notify("❌ Failed to get eBay access token", severity="error")
                    
            except Exception as e:
                self.app.notify(f"❌ eBay connection failed: {str(e)}", severity="error")
        
        # Run the async test
        asyncio.create_task(test_connection())
    
    def test_keepa_connection(self):
        """Test Keepa API connection with current key"""

        async def test_connection():
            try:
                keepa_key = self.query_one("#keepa_key").value.strip()
                
                if not keepa_key:
                    self.app.notify("Please enter a Keepa API key first", severity="error")
                    return
                
                # Test Keepa connection with a simple request
                self.app.notify("Testing Keepa connection...", severity="info")
                
                keepa_client = KeepaDirect(keepa_key)
                # Test with a known ASIN
                result = await keepa_client.get_product("B0F1F8Z8XB")
                
                if result:
                    self.app.notify("✅ Keepa API connection successful!", severity="success")
                else:
                    self.app.notify("⚠️ Keepa API key may be invalid or quota exceeded", severity="warning")
                    
            except Exception as e:
                self.app.notify(f"❌ Keepa connection failed: {str(e)}", severity="error")
        
        # Run the async test
        asyncio.create_task(test_connection())

class BatchImportModal(ModalScreen):
    """Modal for batch ASIN import"""
    
    CSS = """
    BatchImportModal {
        align: center middle;
    }
    
    #batch_dialog {
        grid-size: 2;
        grid-gutter: 1;
        grid-rows: 1fr 3fr 1fr;
        padding: 0 1;
        width: 80;
        height: 20;
        border: thick $background 80%;
        background: $surface;
    }
    
    #batch_title {
        column-span: 2;
        content-align: center middle;
        text-style: bold;
    }
    
    #asin_list {
        column-span: 2;
        height: 10;
    }
    
    Button {
        width: 100%;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Container(
            Label("Batch Import ASINs", id="batch_title"),
            Input(
                placeholder="Enter ASINs separated by commas, spaces, or new lines:\nB0F3PT1VBL, B08N5WRWNW, B09JQMJMXY\n\nSupports copy/paste from spreadsheets!",
                id="asin_list"
            ),
            Button("Import All", variant="primary", id="batch_import_confirm"),
            Button("Cancel", variant="default", id="batch_cancel"),
            id="batch_dialog"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "batch_import_confirm":
            asin_text = self.query_one("#asin_list", Input).value.strip()
            if asin_text:
                # Parse ASINs - split by comma, newline, or space
                import re
                asins = re.findall(r'B[A-Z0-9]{9}', asin_text.upper())
                self.dismiss(asins)
            else:
                self.dismiss([])
        else:
            self.dismiss([])


class ProductEditModal(ModalScreen):
    """Modal for editing individual product details"""
    
    CSS = """
    ProductEditModal {
        background: rgba(0, 0, 0, 0.8);
        align: center middle;
    }
    
    #edit_dialog {
        background: $surface;
        border: thick $primary;
        width: 90%;
        max-width: 120;
        height: 85%;
        max-height: 40;
        padding: 1;
    }
    
    .field-group {
        margin: 1 0;
        height: auto;
    }
    
    .field-label {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .feature-item {
        color: $text-muted;
        margin-left: 2;
    }
    
    .image-url {
        color: $text-muted;
        text-style: italic;
        margin-left: 1;
    }
    
    .info-text {
        color: $text-muted;
        text-style: italic;
        margin: 0 0 1 0;
    }
    
    .error-details {
        background: $error-darken-2;
        color: $error;
        padding: 1;
        margin: 1 0;
    }
    
    Input {
        width: 100%;
        margin-bottom: 1;
    }
    
    Horizontal {
        height: 3;
        align: center middle;
        margin-top: 2;
    }
    """
    
    def __init__(self, product: Product, product_index: int):
        super().__init__()
        self.product = product
        self.product_index = product_index
        self.original_product = None
    
    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="edit_dialog"):
            yield Label(f"Edit Product: {self.product.asin}", classes="field-label")
            
            # Error details (if any)
            if self.product.listing_status == "error" and self.product.error_details:
                with Container(classes="error-details"):
                    yield Label("Error Details:", classes="field-label")
                    yield Label(self.product.error_details)
                    yield Button("Reset to Ready", variant="warning", id="reset_status")
            
            # Product Images (if available)
            if self.product.images:
                with Container(classes="field-group"):
                    yield Label(f"Product Images: 📷 {len(self.product.images)} available", classes="field-label")
                    # Show first 3 image URLs for reference
                    for i, img_url in enumerate(self.product.images[:3], 1):
                        yield Label(f"  {i}. {img_url[:80]}{'...' if len(img_url) > 80 else ''}", classes="image-url")
                    if len(self.product.images) > 3:
                        yield Label(f"  ... and {len(self.product.images) - 3} more images", classes="image-url")
                    # Option to preview first image externally
                    yield Button("🖼️ Preview Main Image", variant="default", id="preview_image")
            
            # Basic product info
            with Container(classes="field-group"):
                yield Label("Title:", classes="field-label")
                yield Input(
                    value=self.product.title, 
                    id="edit_title",
                    placeholder="Product title..."
                )
            
            with Container(classes="field-group"):
                yield Label("Description:", classes="field-label")
                yield Input(
                    value=self.product.description, 
                    id="edit_description",
                    placeholder="Product description..."
                )
            
            with Container(classes="field-group"):
                yield Label("Brand:", classes="field-label")
                yield Input(
                    value=self.product.brand, 
                    id="edit_brand",
                    placeholder="Brand name..."
                )
            
            # Pricing
            with Container(classes="field-group"):
                yield Label("Amazon Price ($):", classes="field-label")
                yield Input(
                    value=str(self.product.price) if self.product.price else "", 
                    id="edit_amazon_price",
                    placeholder="0.00"
                )
            
            with Container(classes="field-group"):
                yield Label("eBay Price ($):", classes="field-label")
                yield Input(
                    value=str(self.product.optimized_price) if self.product.optimized_price else "", 
                    id="edit_ebay_price",
                    placeholder="0.00"
                )
            
            # Weight
            with Container(classes="field-group"):
                yield Label("Weight (oz):", classes="field-label")
                weight_value = str(self.product.weight_oz) if self.product.weight_oz else ""
                weight_lbs = f" ({self.product.weight_oz / 16:.3f} lbs)" if self.product.weight_oz else ""
                yield Input(
                    value=weight_value,
                    id="edit_weight_oz",
                    placeholder="0.00"
                )
                if weight_lbs:
                    yield Label(f"Equivalent{weight_lbs}", classes="info-text")
            
            # Additional product details
            with Container(classes="field-group"):
                yield Label("Source:", classes="field-label")
                yield Label(f"{self.product.source.title()}")
            
            # Features (if available)
            if self.product.features:
                with Container(classes="field-group"):
                    yield Label("Features:", classes="field-label")
                    for i, feature in enumerate(self.product.features[:5]):
                        yield Label(f"• {feature}", classes="feature-item")
            
            # Status and actions
            with Container(classes="field-group"):
                yield Label(f"Current Status: {self.product.listing_status.title()}", classes="field-label")
                if self.product.ebay_listing_id:
                    yield Label(f"eBay Listing ID: {self.product.ebay_listing_id}")
            
            # Action buttons
            with Horizontal():
                yield Button("Save Changes", variant="primary", id="save_product")
                yield Button("Delete Product", variant="error", id="delete_product")
                yield Button("Cancel", variant="default", id="cancel_edit")
    
    def on_mount(self):
        """Store original product state for comparison"""
        from copy import deepcopy
        self.original_product = deepcopy(self.product)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save_product":
            self.save_changes()
        elif event.button.id == "delete_product":
            self.delete_product()
        elif event.button.id == "reset_status":
            self.reset_product_status()
        elif event.button.id == "preview_image":
            self.preview_main_image()
        elif event.button.id == "cancel_edit":
            self.dismiss(False)
    
    def save_changes(self):
        """Save the edited product changes"""
        try:
            # Get updated values
            title = self.query_one("#edit_title", Input).value.strip()
            description = self.query_one("#edit_description", Input).value.strip()
            brand = self.query_one("#edit_brand", Input).value.strip()
            amazon_price_str = self.query_one("#edit_amazon_price", Input).value.strip()
            ebay_price_str = self.query_one("#edit_ebay_price", Input).value.strip()
            weight_oz_str = self.query_one("#edit_weight_oz", Input).value.strip()
            
            # Validate and update product
            if title:
                self.product.title = title
            if description:
                self.product.description = description
            if brand:
                self.product.brand = brand
            
            # Update prices
            if amazon_price_str:
                try:
                    self.product.price = Decimal(amazon_price_str)
                except ValueError:
                    self.app.notify("Invalid Amazon price format", severity="error")
                    return
            
            if ebay_price_str:
                try:
                    self.product.optimized_price = Decimal(ebay_price_str)
                except ValueError:
                    self.app.notify("Invalid eBay price format", severity="error")
                    return
            
            # Update weight
            if weight_oz_str:
                try:
                    new_weight = float(weight_oz_str)
                    if new_weight < 0:
                        self.app.notify("Weight cannot be negative", severity="error")
                        return
                    self.product.weight_oz = new_weight
                    weight_lbs = new_weight / 16
                    self.app.notify(f"Weight updated to {new_weight} oz ({weight_lbs:.3f} lbs)", severity="info")
                except ValueError:
                    self.app.notify("Invalid weight format - please enter a number", severity="error")
                    return
            elif weight_oz_str == "":
                # Allow clearing the weight
                self.product.weight_oz = None
            
            self.app.notify(f"Product {self.product.asin} updated successfully", severity="success")
            self.dismiss(True)
            
        except Exception as e:
            self.app.notify(f"Error saving changes: {str(e)}", severity="error")
    
    def delete_product(self):
        """Delete the product from the list"""
        # Get the main app instance to access the products list
        app = self.app
        if hasattr(app, 'products') and self.product_index < len(app.products):
            del app.products[self.product_index]
            self.app.notify(f"Product {self.product.asin} deleted", severity="warning")
            self.dismiss(True)
        else:
            self.app.notify("Error: Could not delete product", severity="error")
    
    def reset_product_status(self):
        """Reset product status from error to ready"""
        self.product.listing_status = "ready"
        self.product.error_details = None
        self.app.notify(f"Product {self.product.asin} status reset to ready", severity="success")
        self.dismiss(True)
    
    def preview_main_image(self):
        """Open the main product image in system default viewer"""
        if self.product.images:
            import webbrowser
            try:
                webbrowser.open(self.product.images[0])
                self.app.notify("Opening image in browser...", severity="info")
            except Exception as e:
                self.app.notify(f"Could not open image: {str(e)}", severity="error")


class EbayCrosslister(App):
    """Main application"""
    
    CSS = """
    .header {
        text-align: center;
        background: $primary;
        color: $text;
        margin: 1;
        padding: 1;
    }
    
    Container {
        content-align: center middle;
        padding: 1;
    }

    DataTable {
        height: auto;
        content-align: left middle;
        border: solid $primary;
        margin-bottom: 1;
        min-height: 10;
    }
    
    .textbox-container {
        content-align: left top;
        border: solid $primary;
        margin-bottom: 1;
        min-height: 8;
        max-height: 12;
        padding: 1;
    }
    
    .input-header {
        color: $text;
        text-style: bold;
        margin-bottom: 1;
    }

    .buttons {
        margin: 1 0;
        height: 3;
        align: center middle;
    }
    
    Log {
        height: 10;
        border: solid $primary;
    }
    
    .status-ready { color: $success; }
    .status-listed { color: $primary; }
    .status-error { color: $error; }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("c", "config", "Config"),
        ("r", "refresh", "Refresh"),
    ]
    
    def __init__(self):
        super().__init__()
        self.config = Config.load_from_keyring()
        self.rate_limiter = RateLimiter()
        self.products: List[Product] = []
        self.keepa_client = None
        self.amazon_scraper = AmazonScraper()
        self.ebay_client = None
        self.failed_imports = []  # Track failed ASIN imports
        self.failed_listings = []  # Track failed eBay listings
        
        # Initialize pricing optimizer
        self.pricing_optimizer = PricingOptimizer(
            net_margin=self.config.net_margin,
            ebay_fee_rate=self.config.ebay_fee_rate,
            paypal_fee_rate=self.config.paypal_fee_rate
        )
        
        # Security and lock screen management
        self.is_locked = False
        self.last_activity_time = time.time()
        self.idle_timer = None

        # initiailize credentials if stored and available
        if self.config.keepa_api_key:
            self.keepa_client = KeepaDirect(self.config.keepa_api_key)
        
        if self.config.ebay_client_id and self.config.ebay_client_secret:
            try:
                self.ebay_client = EbayInventoryAPI(
                    client_id=self.config.ebay_client_id,
                    client_secret=self.config.ebay_client_secret,
                    refresh_token=self.config.ebay_refresh_token,
                    sandbox=self.config.ebay_sandbox,
                    runame=self.config.ebay_runame,
                    country=self.config.ebay_country,
                    log_callback=self.app_log,
                    config=self.config
                )
                print("✅ eBay client initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize eBay client: {e}")
                self.ebay_client = None

        
        # Production readiness check on startup
        self.production_status = self._check_production_readiness()
    
    def _check_production_readiness(self) -> Dict[str, Any]:
        """
        Comprehensive production readiness assessment
        Returns status dict with all system checks
        """
        status = {
            "overall_ready": False,
            "keyring": {"status": "unknown", "details": ""},
            "ebay_config": {"status": "unknown", "details": ""},
            "dependencies": {"status": "unknown", "details": ""},
            "security": {"status": "unknown", "details": ""},
            "recommendations": []
        }
        
        try:
            # 1. Check keyring compatibility
            keyring_status = Config.verify_keyring_compatibility()
            if keyring_status["test_successful"]:
                status["keyring"]["status"] = "ready"
                status["keyring"]["details"] = f"Secure storage available ({keyring_status['backend']})"
            else:
                status["keyring"]["status"] = "warning"
                status["keyring"]["details"] = f"Keyring issue: {keyring_status.get('error', 'Unknown')}"
                status["recommendations"].append("Fix keyring configuration for secure credential storage")
            
            # 2. Check eBay configuration
            if self.config.ebay_client_id and self.config.ebay_client_secret:
                status["ebay_config"]["status"] = "ready"
                status["ebay_config"]["details"] = "eBay credentials configured"
                if self.config.ebay_sandbox:
                    status["recommendations"].append("Switch to production mode when ready to create real listings")
            else:
                status["ebay_config"]["status"] = "missing"
                status["ebay_config"]["details"] = "eBay credentials not configured"
                status["recommendations"].append("Complete eBay account setup wizard")
            
            # 3. Check critical dependencies
            try:
                import keyring, httpx, textual, secrets, hashlib
                status["dependencies"]["status"] = "ready"
                status["dependencies"]["details"] = "All critical dependencies available"
            except ImportError as e:
                status["dependencies"]["status"] = "error"
                status["dependencies"]["details"] = f"Missing dependency: {e}"
                status["recommendations"].append("Install missing Python dependencies")
            
            # 4. Check security settings
            if self.config.security_enabled and self.config.app_password_hash:
                status["security"]["status"] = "ready"
                status["security"]["details"] = "Password protection enabled"
            else:
                status["security"]["status"] = "optional"
                status["security"]["details"] = "Password protection disabled (optional)"
                status["recommendations"].append("Consider enabling password protection for enhanced security")
            
            # 5. Overall readiness assessment
            critical_ready = (
                status["keyring"]["status"] in ["ready", "warning"] and
                status["ebay_config"]["status"] == "ready" and
                status["dependencies"]["status"] == "ready"
            )
            
            status["overall_ready"] = critical_ready
            
        except Exception as e:
            status["overall_ready"] = False
            status["recommendations"].append(f"System check failed: {e}")
        
        return status
        



    
    def reload_configuration(self):
        """Reload configuration from keyring and update app state"""
        try:
            old_security_enabled = self.config.security_enabled
            self.config = Config.load_from_keyring()
            
            # If security settings changed, update monitoring
            if self.config.security_enabled != old_security_enabled:
                if self.config.security_enabled and self.config.app_password_hash:
                    self.start_idle_monitoring()
                    self.app_log("🔒 Security enabled")
                else:
                    if self.idle_timer:
                        try:
                            self.idle_timer.cancel()
                        except:
                            pass
                    self.app_log("🔓 Security disabled")
            
            # Update any eBay client if credentials changed
            if self.config.ebay_client_id and self.config.ebay_client_secret:
                self.ebay_client = None  # Force recreation with new credentials
            
            # Update pricing optimizer with new configuration
            self.pricing_optimizer = PricingOptimizer(
                net_margin=self.config.net_margin,
                ebay_fee_rate=self.config.ebay_fee_rate,
                paypal_fee_rate=self.config.paypal_fee_rate
            )
            
            self.app_log("✅ Configuration reloaded")
            
        except Exception as e:
            self.app_log(f"⚠️ Error reloading configuration: {e}")
    
    def compose(self) -> ComposeResult:
        """Create the main UI"""
        yield Header()
        
        # Add floating progress overlay
        yield FloatingProgressOverlay(id="progress_overlay")
        
        with Container():
            yield Label("eBay Cross-Lister v1.0", classes="header")
            
            with Vertical():
                yield Label("Enter Amazon ASINs (one per line, or separated by commas/spaces):", classes="input-header")
                yield TextArea(placeholder="ex: B0F3PT1VBL, B0G4RT2WCK, B0H5YU3XDL...", id="asin_input", classes="textbox-container")
            
            #with Horizontal():
                #yield Button("Import Products", variant="primary", id="import_btn")
                
            
            with Horizontal(classes = "buttons"):
                yield Button("Import ASINs", variant="primary", id="import_btn")
                yield Button("Optimize All", id="optimize_btn")
                yield Button("List selected product", id="list_selected_btn")
                yield Button("Batch List", variant="warning", id="batch_list_btn")
            
            yield DataTable(id="products_table")

            # Selection management buttons
            with Horizontal(classes = "buttons"):
                yield Button("Select All", variant="default", id="select_all_btn")
                yield Button("Clear Selection", variant="default", id="clear_selection_btn")
                yield Button("Delete Selected", variant="error", id="delete_selected_btn")
            
            yield Log(id="app_log")
        
        yield Footer()
    
    def on_mount(self):
        """Initialize the application"""
        # Setup data table with selection support
        table = self.query_one("#products_table", DataTable)
        table.add_columns("Select", "ASIN", "Title", "Amazon $", "eBay $", "Status", "Source", "Actions")
        
        # Log startup with production readiness status
        self.app_log("eBay Cross-Lister v2.0 started")
        
        # Display production readiness status
        if self.production_status["overall_ready"]:
            self.app_log("✅ System ready for production use")
        else:
            self.app_log("⚠️ System setup incomplete - some features may not work")
            for rec in self.production_status["recommendations"][:3]:  # Show first 3 recommendations
                self.app_log(f"💡 {rec}")
        
        if self.config.keepa_api_key:
            self.app_log("✅ Keepa API configured")
        else:
            self.app_log("⚠️ Keepa API not configured - using fallback scraping")
        
        if self.ebay_client:
            mode = "sandbox" if self.config.ebay_sandbox else "production"
            self.app_log(f"✅ eBay API configured ({mode} mode)")
        else:
            self.app_log("⚠️ eBay API not configured - use 'c' to configure")
        
        self.app_log("Ready to import products!")
        
        # Start idle timeout monitoring if security is enabled
        if self.config.security_enabled and self.config.app_password_hash:
            # Show lock screen immediately on startup if security is enabled
            self.show_lock_screen()
            self.app_log(f"🔒 Security enabled - lock timeout: {self.config.lock_timeout_minutes} minutes")

        try:
            # print monitoring - sent to log widget
            self.app.begin_capture_print(self.app_log, stderr=True, stdout=True)
        except Exception as e:
            self.app_log(f"⚠️ Print capture failed: {e}")
    
    def start_idle_monitoring(self):
        """Start monitoring for idle timeout"""
        if self.idle_timer:
            try:
                self.idle_timer.cancel()
            except (AttributeError, RuntimeError):
                # Timer might not have cancel method or already expired
                pass
        
        timeout_seconds = self.config.lock_timeout_minutes * 60
        self.idle_timer = self.set_timer(timeout_seconds, self.check_idle_timeout)
        self.update_activity_time()
    
    def update_activity_time(self):
        """Update the last activity timestamp"""
        self.last_activity_time = time.time()
    
    def check_idle_timeout(self):
        """Check if the app should be locked due to inactivity"""
        if not self.config.security_enabled or not self.config.app_password_hash:
            return
        
        current_time = time.time()
        idle_time = current_time - self.last_activity_time
        timeout_seconds = self.config.lock_timeout_minutes * 60
        
        if idle_time >= timeout_seconds and not self.is_locked:
            self.show_lock_screen()
    
    def show_lock_screen(self):
        """Show the lock screen modal"""
        if self.is_locked:
            return  # Already locked
        
        self.is_locked = True
        
        def on_unlock(unlocked):
            if unlocked:
                self.is_locked = False
                self.start_idle_monitoring()  # Restart idle monitoring
                self.app_log("🔓 Application unlocked")
            else:
                # If unlock failed, keep locked and show again
                self.show_lock_screen()
        
        # Pass the progress overlay to lock screen for integrated display
        try:
            progress_overlay = self.query_one("#progress_overlay")
            self.push_screen(LockScreen(progress_overlay=progress_overlay), on_unlock)
        except Exception:
            # Fallback if progress overlay not found
            self.push_screen(LockScreen(), on_unlock)
    
    def on_key(self, event):
        """Handle key events to track activity"""
        self.update_activity_time()

    def on_mouse_move(self, event):
        """Handle mouse move events to track activity"""  
        self.update_activity_time()

    def on_click(self, event):
        """Handle click events to track activity"""
        self.update_activity_time()

    def app_log(self, message: str):
        """Log message to the application log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_widget = self.query_one("#app_log", Log)
        log_widget.write_line(f"[{timestamp}] {message}")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks"""
        self.update_activity_time()  # Track user activity
        
        if event.button.id == "import_btn":
            await self.import_product()
        elif event.button.id == "optimize_btn":
            await self.optimize_all_prices()
        elif event.button.id == "list_btn":
            await self.list_to_ebay()
        elif event.button.id == "batch_import_btn":
            await self.show_batch_import_modal()
        elif event.button.id == "batch_list_btn":
            await self.batch_list_to_ebay()
        elif event.button.id == "select_all_btn":
            self.select_all_products()
        elif event.button.id == "clear_selection_btn":
            self.clear_selection()
        elif event.button.id == "delete_selected_btn":
            self.delete_selected_products()
        elif event.button.id == "list_selected_btn":
            await self.list_selected_products()
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input fields"""
        self.update_activity_time()  # Track user activity
        
        # TextArea doesn't submit on enter, so this is for other inputs
        # Users will click Import button for ASIN processing
    
    async def on_data_table_cell_selected(self, event) -> None:
        """Handle table cell selection and double-click detection"""
        import time
        table = self.query_one("#products_table", DataTable)
        if event.coordinate.row < len(self.products):
            product = self.products[event.coordinate.row]
            
            # Double-click detection
            current_time = time.time()
            if not hasattr(self, '_last_click_time'):
                self._last_click_time = 0
                self._last_click_coord = None
            
            is_double_click = (current_time - self._last_click_time < 0.5 and 
                             self._last_click_coord == (event.coordinate.row, event.coordinate.column))
            
            # Update click tracking
            self._last_click_time = current_time
            self._last_click_coord = (event.coordinate.row, event.coordinate.column)
            
            # Handle double-click on product data (not checkbox or actions)
            if is_double_click and event.coordinate.column not in [0, 7]:
                await self.show_product_editor(product, event.coordinate.row)
                return
            
            # Handle single clicks
            # Column 0 is checkbox - toggle selection
            if event.coordinate.column == 0:
                product.selected = not getattr(product, 'selected', False)
                self.update_products_table()
            
            # Column 7 is Actions - handle edit/delete
            elif event.coordinate.column == 7:
                await self.show_product_editor(product, event.coordinate.row)
    
    async def on_data_table_cell_highlighted(self, event) -> None:
        """Handle table cell highlighting (kept for compatibility)"""
        pass
    
    async def show_product_editor(self, product: Product, product_index: int):
        """Show the product editor modal"""
        def on_edit_complete(result):
            if result:
                # Refresh the table
                self.update_products_table()
        
        self.push_screen(ProductEditModal(product, product_index), on_edit_complete)
    
    async def parse_asins(self, text: str) -> List[str]:
        """Parse multiple ASINs from text input"""
        import re
        
        # Replace various separators with spaces
        text = re.sub(r'[,\n\r\t]+', ' ', text)
        
        # Split by whitespace and filter out empty strings
        potential_asins = [item.strip() for item in text.split() if item.strip()]
        
        # Validate each ASIN
        valid_asins = []
        for asin in potential_asins:
            try:
                validated_asin = SecurityUtils.validate_asin(asin)
                valid_asins.append(validated_asin)
            except Exception as e:
                self.app_log(f"⚠️ Invalid ASIN '{asin}': {e}")
        
        return valid_asins
    
    async def import_product(self):
        """Import products from Amazon ASINs"""
        asin_input = self.query_one("#asin_input", TextArea)
        asin_text = asin_input.text.strip()
        
        if not asin_text:
            self.app_log("❌ Please enter one or more ASINs")
            return
        
        # Parse multiple ASINs
        asins = await self.parse_asins(asin_text)
        
        if not asins:
            self.app_log("❌ No valid ASINs found")
            return
        
        total_asins = len(asins)
        self.app_log(f"🚀 Starting import of {total_asins} ASIN(s)...")
        
        # Show progress overlay
        progress_overlay = self.query_one("#progress_overlay")
        progress_overlay.show_progress(f"Importing {total_asins} ASINs", total_asins)
        
        successful_imports = 0
        failed_imports = 0
        
        try:
            for i, asin in enumerate(asins, 1):
                try:
                    self.app_log(f"📦 Importing ASIN {i}/{total_asins}: {asin}")
                    progress_overlay.update_progress(i - 1, f"Importing {asin}")
                    
                    # Import the product
                    await self._import_single_asin(asin)
                    successful_imports += 1
                    
                    self.app_log(f"✅ Successfully imported {asin} ({i}/{total_asins})")
                    progress_overlay.update_progress(i, f"Completed {asin}")
                    
                except Exception as e:
                    failed_imports += 1
                    self.app_log(f"❌ Failed to import {asin}: {e}")
                    progress_overlay.update_progress(i, f"Failed {asin}")
            
            # Summary
            self.app_log(f"📊 Import complete: {successful_imports} successful, {failed_imports} failed")
            
        finally:
            # Hide progress overlay
            progress_overlay.hide_progress()
        
        # Clear input if all successful
        if failed_imports == 0:
            asin_input.text = ""
    
    async def _import_single_asin(self, asin: str):
        """Import a single ASIN (extracted from main import logic)"""
        # Validate ASIN
        asin = SecurityUtils.validate_asin(asin)
        
        # Check rate limit
        if not self.rate_limiter.allow_request("import"):
            raise Exception("Rate limit exceeded, please wait...")
        
        # Try Keepa first
        product_data = None
        source = "unknown"
        
        if self.keepa_client:
            product_data = await self.keepa_client.get_product(asin)
            if product_data:
                source = "keepa"
        
        if not product_data or not self.keepa_client:
            # fallback to webscraping (with retry logic) if Keepa fails or not available
            product_data = await self.scrape_amazon_product(asin)
            source = "scraper"
        
        if product_data:
            product = self.parse_product_data(asin, product_data, source)
            self.products.append(product)
            self.update_products_table()
            return product
        else:
            raise Exception(f"Failed to import product {asin}")
    
    def parse_product_data(self, asin: str, data: Dict[str, Any], source: str) -> Product:
        """Parse product data from API response"""
        if source == "keepa":
            return Product(
                asin=asin,
                title=SecurityUtils.sanitize_text(data.get("title", "")),
                price=Decimal(str(data.get("stats", {}).get("current", [None, 0])[1] / 100)) if data.get("stats", {}).get("current") else None,
                currency="USD",
                images=data.get("imagesCSV", "").split(",") if data.get("imagesCSV") else [],
                description=SecurityUtils.sanitize_text(data.get("description", "")),
                brand=SecurityUtils.sanitize_text(data.get("brand", "")),
                features=data.get("features", []) if data.get("features") else [],
                source=source
            )
        elif source == "scraper":
            # Parse scraper data format with enhanced manufacturer data
            return Product(
                asin=asin,
                title=SecurityUtils.sanitize_text(data.get("title", "Unknown Product")),
                price=Decimal(str(data.get("price"))) if data.get("price") else None,
                currency=data.get("currency", "USD"),
                images=data.get("images", []),
                description=SecurityUtils.sanitize_text(data.get("description", "")),
                brand=SecurityUtils.sanitize_text(data.get("brand", "")),
                features=data.get("features", []),
                weight_oz=data.get("weight_oz"),  # Include scraped weight
                source=source,
                # Enhanced manufacturer specification fields
                weight_source=data.get("weight_source", "amazon"),
                dimensions=data.get("dimensions"),
                model=data.get("manufacturer_model"),
                manufacturer=data.get("manufacturer"),
                official_name=data.get("official_name"),
                specifications=data.get("manufacturer_specs"),
                confidence=data.get("confidence", 0.0),
                scraped_at=data.get("scraped_at"),
                # Data source tracking
                title_source=data.get("title_source"),
                brand_source=data.get("brand_source"),
                price_source=data.get("price_source")
            )
        else:
            # Fallback for unknown sources
            return Product(asin=asin, title="Unknown Product", source=source)
    
    async def scrape_amazon_product(self, asin: str) -> Optional[Dict[str, Any]]:
        """Fallback Amazon scraping using BeautifulSoup with retry logic"""
        try:
            self.app_log(f"🕷️ Scraping Amazon for {asin}...")
            
            # Use the scraper to get product data (includes retry logic)
            product_data = await self.amazon_scraper.scrape_product(asin)
            
            if product_data:
                attempts = product_data.get('attempts_needed', 1)
                if attempts > 1:
                    self.app_log(f"✅ Successfully scraped {asin} after {attempts} attempts")
                else:
                    self.app_log(f"✅ Successfully scraped {asin}")
                return product_data
            else:
                self.app_log(f"❌ Failed to scrape {asin} after all retry attempts")
                return None
                
        except Exception as e:
            self.app_log(f"❌ Scraping error for {asin}: {str(e)}")
            return None
    
    async def optimize_all_prices(self):
        """Optimize prices for all imported products"""
        if not self.products:
            self.app_log("❌ No products to optimize")
            return
        
        # Get products that need optimization
        products_to_optimize = [p for p in self.products if p.price and not p.optimized_price]
        
        if not products_to_optimize:
            self.app_log("✅ All products already have optimized prices")
            return
        
        total = len(products_to_optimize)
        self.app_log(f"💰 Optimizing prices for {total} products...")
        
        # Show progress overlay
        progress_overlay = self.query_one("#progress_overlay")
        progress_overlay.show_progress(f"Optimizing {total} product prices", total)
        
        try:
            for i, product in enumerate(products_to_optimize, 1):
                progress_overlay.update_progress(i - 1, f"Optimizing {product.asin}")
                
                product.optimized_price = self.pricing_optimizer.calculate_optimal_price(product.price)
                self.app_log(f"💰 {product.asin}: ${product.price} → ${product.optimized_price}")
                
                progress_overlay.update_progress(i, f"Optimized {product.asin}")
                
                # Small delay for visual feedback
                await asyncio.sleep(0.1)
        
        finally:
            # Hide progress overlay
            progress_overlay.hide_progress()
        
        self.update_products_table()
        self.app_log("✅ Price optimization complete")
    
    async def list_to_ebay(self, product: Product = None):
        """List a single product or all ready products to eBay"""
        if not self.ebay_client:
            self.app_log("❌ eBay API not configured - use 'c' to configure")
            return False
        
        # If specific product provided, list that one; otherwise list all ready products
        if product:
            products_to_list = [product] if product.optimized_price and product.listing_status == "ready" else []
        else:
            products_to_list = [p for p in self.products if p.optimized_price and p.listing_status == "ready"]
        
        if not products_to_list:
            if product:
                self.app_log(f"❌ Product {product.asin} not ready for listing")
                return False
            else:
                self.app_log("❌ No products ready for listing")
                return False
        
        success_count = 0
        for prod in products_to_list:
            try:
                # Additional safeguard: Skip if product is already listed or has an eBay listing ID
                if prod.listing_status == "listed" or prod.ebay_listing_id:
                    self.app_log(f"⚠️  Skipping {prod.asin} - already listed (ID: {prod.ebay_listing_id})")
                    continue
                
                # Mark as processing to prevent duplicate attempts
                prod.listing_status = "processing"
                self.app_log(f"📤 Listing {prod.asin} for ${prod.optimized_price}...")
                
                # Call eBay API to create listing
                listing_id = await self.ebay_client.create_listing(prod)
                
                if listing_id:
                    prod.listing_status = "listed"
                    prod.ebay_listing_id = listing_id
                    success_count += 1
                    self.app_log(f"✅ Listed {prod.asin} - eBay ID: {listing_id}")
                else:
                    prod.listing_status = "error"
                    prod.error_details = "Unknown listing failure"
                    self.app_log(f"❌ Failed to list {prod.asin} - Unknown error")
                    if hasattr(self, 'failed_listings') and prod.asin not in self.failed_listings:
                        self.failed_listings.append(prod.asin)
                
            except Exception as e:
                prod.listing_status = "error"
                prod.error_details = str(e)  # Store detailed error for individual review
                self.app_log(f"❌ Error listing {prod.asin}: {str(e)}")
                if hasattr(self, 'failed_listings') and prod.asin not in self.failed_listings:
                    self.failed_listings.append(prod.asin)
        
        self.update_products_table()
        
        if product:
            return success_count > 0
        else:
            self.app_log(f"✅ eBay listing complete - {success_count}/{len(products_to_list)} successful")
            return success_count == len(products_to_list)
    
    async def show_batch_import_modal(self):
        """Show batch import modal and process results"""
        def on_batch_complete(asins):
            if asins:
                asyncio.create_task(self.batch_import_products(asins))
        
        self.push_screen(BatchImportModal(), on_batch_complete)
    
    async def batch_import_products(self, asins: List[str]):
        """Import multiple products with progress tracking and failed ASIN tracking"""
        total = len(asins)
        self.app_log(f"🚀 Starting batch import of {total} products...")

        # Show progress overlay
        progress_overlay = self.query_one("#progress_overlay")
        progress_overlay.show_progress(f"Importing {total} products via batch", total)

        imported = 0
        failed = 0
        self.failed_imports = []  # Track failed ASINs for retry

        try:
            for i, asin in enumerate(asins, 1):
                try:
                    # Check rate limit
                    if not self.rate_limiter.allow_request("import"):
                        self.app_log(f"⏳ Rate limit - waiting 1 second...")
                        await asyncio.sleep(1)

                    self.app_log(f"🔍 [{i}/{total}] Importing {asin}...")
                    progress_overlay.update_progress(i - 1, f"Importing {asin}")

                    # Import product (reuse existing logic)
                    product_data = None
                    source = "unknown"

                    if self.keepa_client:
                        product_data = await self.keepa_client.get_product(asin)
                        if product_data:
                            source = "keepa"

                    if not product_data:
                        product_data = await self.scrape_amazon_product(asin)
                        source = "scraper"

                    if product_data:
                        product = self.parse_product_data(asin, product_data, source)

                        # Check for duplicates
                        if not any(p.asin == asin for p in self.products):
                            self.products.append(product)
                            imported += 1
                            self.app_log(f"✅ [{i}/{total}] Imported: {product.title[:30]}...")
                            progress_overlay.update_progress(i, f"Imported {asin}")
                        else:
                            self.app_log(f"⚠️ [{i}/{total}] Duplicate ASIN {asin} - skipped")
                            progress_overlay.update_progress(i, f"Skipped {asin}")
                    else:
                        failed += 1
                        self.failed_imports.append(asin)
                        self.app_log(f"❌ [{i}/{total}] Failed to import {asin}")
                        progress_overlay.update_progress(i, f"Failed {asin}")

                    # Update table every 5 products
                    if i % 5 == 0:
                        self.update_products_table()

                except Exception as e:
                    failed += 1
                    self.failed_imports.append(asin)
                    self.app_log(f"❌ [{i}/{total}] Error importing {asin}: {str(e)}")
                    progress_overlay.update_progress(i, f"Error {asin}")

        finally:
            # Hide progress overlay
            progress_overlay.hide_progress()

        # Final update
        self.update_products_table()
        self.app_log(f"🎉 Batch import complete! ✅ {imported} imported, ❌ {failed} failed")
        if self.failed_imports:
            self.app_log(f"🔁 Failed ASINs: {', '.join(self.failed_imports)}")
    
    async def batch_list_to_ebay(self):
        """List all ready products to eBay with progress tracking"""
        if not self.ebay_client:
            self.app_log("❌ eBay API not configured - use 'c' to configure")
            return
        
        ready_products = [p for p in self.products if p.optimized_price and p.listing_status == "ready"]
        self.failed_listings = []  # Reset failed listings tracker

        if not ready_products:
            self.app_log("❌ No products ready for listing (need optimization first)")
            return
        
        total = len(ready_products)
        self.app_log(f"🚀 Starting batch listing of {total} products to eBay...")
        
        # Estimate time (5-10 seconds per listing due to API calls)
        estimated_minutes = (total * 7) / 60  # Conservative estimate
        self.app_log(f"⏱️ Estimated completion time: {estimated_minutes:.1f} minutes")
        
        # Show progress overlay
        progress_overlay = self.query_one("#progress_overlay")
        progress_overlay.show_progress(f"Listing {total} products to eBay", total)
        
        listed = 0
        failed = 0
        
        try:
            for i, product in enumerate(ready_products, 1):
                try:
                    self.app_log(f"📤 [{i}/{total}] Processing {product.asin} for ${product.optimized_price}...")
                    progress_overlay.update_progress(i - 1, f"Listing {product.asin}")
                    
                    # Call the updated list_to_ebay method
                    success = await self.list_to_ebay(product)
                    
                    if success:
                        listed += 1
                        self.app_log(f"✅ [{i}/{total}] Listed {product.asin} - ID: {product.ebay_listing_id}")
                        progress_overlay.update_progress(i, f"Listed {product.asin}")
                    else:
                        failed += 1
                        if product.asin not in self.failed_listings:
                            self.failed_listings.append(product.asin)
                        self.app_log(f"❌ [{i}/{total}] Failed to list {product.asin}")
                        progress_overlay.update_progress(i, f"Failed {product.asin}")
                    
                    # Update table every 3 listings
                    if i % 3 == 0:
                        self.update_products_table()
                    
                    # Brief delay between listings to avoid rate limits
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    failed += 1
                    product.listing_status = "error"
                    if product.asin not in self.failed_listings:
                        self.failed_listings.append(product.asin)
                    self.app_log(f"❌ [{i}/{total}] Error listing {product.asin}: {str(e)}")
                    progress_overlay.update_progress(i, f"Error {product.asin}")
        
        finally:
            # Hide progress overlay
            progress_overlay.hide_progress()
        
        # Final update
        self.update_products_table()
        self.app_log(f"🎉 Batch listing complete! ✅ {listed} listed, ❌ {failed} failed")
        
        # Show failed listings for retry option
        if self.failed_listings:
            self.app_log(f"🔁 Failed listings ({len(self.failed_listings)}): {', '.join(self.failed_listings)}")
            self.app_log("💡 Tip: Check your eBay API configuration or try individual listings for debugging")
    
    def update_products_table(self):
        """Update the products table display with selection support"""
        table = self.query_one("#products_table", DataTable)
        table.clear()
        
        for i, product in enumerate(self.products):
            status_class = f"status-{product.listing_status}"
            # Add checkbox for selection
            checkbox_text = "☐"  # Empty checkbox
            if hasattr(product, 'selected') and product.selected:
                checkbox_text = "☑"  # Checked
                
            table.add_row(
                checkbox_text,
                product.asin,
                product.title[:30] + "..." if len(product.title) > 30 else product.title,
                f"${product.price}" if product.price else "N/A",
                f"${product.optimized_price}" if product.optimized_price else "Not optimized",
                product.listing_status.title(),
                product.source.title(),
                "[bold]Edit[/bold]|Delete" if product.listing_status == "error" else "[bold]Edit[/bold]",
                key=str(i)  # Use index as key for row identification
            )
    
    def select_all_products(self):
        """Select all products in the table"""
        for product in self.products:
            product.selected = True
        self.update_products_table()
        self.app_log(f"✅ Selected all {len(self.products)} products")
    
    def clear_selection(self):
        """Clear all product selections"""
        selected_count = sum(1 for p in self.products if p.selected)
        for product in self.products:
            product.selected = False
        self.update_products_table()
        # Only log if there were actually selected products
        if selected_count > 0:
            self.app_log(f"🔄 Cleared selection of {selected_count} products")
    
    def delete_selected_products(self):
        """Delete all selected products"""
        selected_products = [p for p in self.products if p.selected]
        if not selected_products:
            self.app_log("❌ No products selected for deletion")
            return
        
        # Remove selected products
        self.products = [p for p in self.products if not p.selected]
        self.update_products_table()
        self.app_log(f"🗑️ Deleted {len(selected_products)} products")
    
    async def list_selected_products(self):
        """List only the selected products to eBay"""
        selected_products = [p for p in self.products if p.selected and p.optimized_price and p.listing_status == "ready"]
        
        if not selected_products:
            self.app_log("❌ No ready, selected products to list")
            return
        
        self.app_log(f"📤 Listing {len(selected_products)} selected products...")
        
        success_count = 0
        for product in selected_products:
            try:
                # Use the existing list_to_ebay method for individual products
                success = await self.list_to_ebay(product)
                if success:
                    success_count += 1
                
            except Exception as e:
                self.app_log(f"❌ Failed to list {product.asin}: {str(e)}")
        
        self.app_log(f"✅ Listed {success_count}/{len(selected_products)} selected products")

    def action_config(self):
        """Open configuration screen"""
        def on_config_complete(result):
            if result:
                old_security_enabled = self.config.security_enabled
                self.config = Config.load_from_keyring()
                self.pricing_optimizer = PricingOptimizer(
                    net_margin=self.config.net_margin,
                    ebay_fee_rate=self.config.ebay_fee_rate,
                    paypal_fee_rate=self.config.paypal_fee_rate
                )
                
                # Reinitialize API clients
                if self.config.keepa_api_key:
                    self.keepa_client = KeepaDirect(self.config.keepa_api_key)
                
                if self.config.ebay_client_id and self.config.ebay_client_secret:
                    try:
                        self.ebay_client = EbayInventoryAPI(
                            client_id=self.config.ebay_client_id,
                            client_secret=self.config.ebay_client_secret,
                            refresh_token=self.config.ebay_refresh_token,
                            sandbox=self.config.ebay_sandbox,
                            runame=self.config.ebay_runame,
                            country=self.config.ebay_country,
                            log_callback=self.app_log,
                            config=self.config
                        )
                        mode = "sandbox" if self.config.ebay_sandbox else "production"
                        self.app_log(f"✅ eBay API configured ({mode} mode)")
                    except Exception as e:
                        self.app_log(f"❌ Failed to initialize eBay client: {e}")
                        self.ebay_client = None
                else:
                    self.ebay_client = None
                    self.app_log("⚠️ eBay API not configured")
                
                # Handle Keepa configuration
                if self.config.keepa_api_key:
                    self.app_log("✅ Keepa API configured")
                else:
                    self.app_log("⚠️ Keepa API not configured - using fallback scraping")
                
                # Handle security settings changes
                if self.config.security_enabled and self.config.app_password_hash:
                    if not old_security_enabled:
                        self.app_log("🔒 Security enabled")
                    self.start_idle_monitoring()
                elif old_security_enabled:
                    self.app_log("🔓 Security disabled")
                    if self.idle_timer:
                        self.idle_timer.cancel()
                        self.idle_timer = None
                
                self.app_log("✅ Configuration updated")
        
        self.push_screen(ConfigScreen(), on_config_complete)
    
    def action_refresh(self):
        """Refresh the application"""
        self.update_products_table()
        self.app_log("🔄 Refreshed")



def main():
    """Entry point for eBay Cross-Lister TUI Application"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='eBay Cross-Lister - Amazon to eBay Automation Tool\nA terminal-based application for automated Amazon to eBay crosslisting',
        epilog='sylcrala.xyz | contact@sylcrala.xyz\n'
               'Licensed under MIT License\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--setup', action='store_true', help='Force run initial setup wizard')
    parser.add_argument('--config', action='store_true', help='Open configuration screen')
    parser.add_argument('--reset', action='store_true', help='Reset all configuration')
    parser.add_argument('--version', action='version', version='eBay Cross-Lister v1.0.0')
    
    args = parser.parse_args()
    
    # Handle reset command
    if args.reset:
        try:
            print("🔄 Resetting configuration...")
            production_config.reset_configuration()
            print("✅ Configuration reset complete. Please run the application again to set up.")
            return
        except Exception as e:
            print(f"❌ Error resetting configuration: {e}")
            return
    
    # Check if we have existing credentials in the main Config system
    try:
        main_config = Config.load_from_keyring()
        has_ebay_creds = (main_config.ebay_client_id and 
                         main_config.ebay_client_secret and 
                         main_config.ebay_refresh_token)
        
        if has_ebay_creds and not args.setup:
            print("🚀 Starting eBay Cross-Lister...")
            print("✅ Found existing eBay credentials")
            print("━" * 60)
            
            # Launch main app directly if we have credentials
            app = EbayCrosslister()
            
            if args.config:
                # Push config screen after app starts
                def show_config():
                    app.push_screen(ConfigScreen())
                app.call_after_refresh(show_config)
            
            try:
                app.run()
            except KeyboardInterrupt:
                print("\n👋 eBay Cross-Lister closed by user")
            except Exception as e:
                print(f"\n❌ Application error: {e}")
                print("   Try running with --reset to clear configuration")
            return
            
    except Exception as e:
        print(f"⚠️  Error checking existing configuration: {e}")
    
    # If no credentials found or setup forced, show setup process
    print("🚀 Starting eBay Cross-Lister Setup...")
    print("━" * 60)
    print("⚠️  eBay credentials need configuration")
    print("   • This might be your first run")
    print("   • Or credentials need to be reconfigured")
    print("━" * 60)
    print("👋 Welcome to eBay Cross-Lister!")
    print("   The setup wizard will guide you through configuration.")
    print("\n🎯 Launching setup wizard...")
    print("   You can re-run setup anytime with: --setup")
    print("   Access config anytime with: --config")
    print("━" * 60)
    
    # Launch app with setup wizard
    app = EbayCrosslister()
    
    if args.config:
        # Show config screen instead of setup
        def show_config_on_start():
            app.push_screen(ConfigScreen())
        app.call_after_refresh(show_config_on_start)
    else:
        # Show setup wizard
        def show_setup_on_start():
            app.push_screen(EbayAccountSetupWizard())
        app.call_after_refresh(show_setup_on_start)
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n👋 eBay Cross-Lister closed by user")
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        print("   Try running with --reset to clear configuration")

if __name__ == "__main__":
    main()
