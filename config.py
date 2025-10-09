#!/usr/bin/env python3

# Copyright (c) 2025 sylcrala | sylcrala.xyz
# Licensed under the MIT License - see LICENSE.md for details

"""
Production Configuration Management for eBay Cross-Lister
Handles secure credential storage, environment management, and setup workflow
"""

import os
import json
import keyring
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

class ProductionConfig:
    """Manages production configuration and credentials securely"""
    
    def __init__(self):
        self.app_name = "eBay-CrossLister"
        self.config_dir = Path.home() / ".ebay_crosslister"
        self.config_file = self.config_dir / "config.json"
        self.setup_completed_file = self.config_dir / "setup_completed"
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging to config directory"""
        log_file = self.config_dir / "app.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def is_first_run(self) -> bool:
        """Check if this is the first run requiring setup"""
        return not self.setup_completed_file.exists()
    
    def mark_setup_complete(self):
        """Mark initial setup as completed"""
        self.setup_completed_file.touch()
        self.logger.info("Initial setup marked as complete")
    
    def needs_reconfiguration(self) -> bool:
        """Check if reconfiguration is needed (missing credentials or config)"""
        try:
            # Check for essential eBay credentials
            ebay_client_id = self.get_credential("ebay", "client_id")
            ebay_client_secret = self.get_credential("ebay", "client_secret")
            
            # Check for config file
            config = self.load_config()
            
            if not ebay_client_id or not ebay_client_secret:
                return True
                
            if not config.get("ebay_configured", False):
                return True
                
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking configuration: {e}")
            return True
    
    def store_credential(self, service: str, username: str, password: str):
        """Securely store credentials using OS keyring"""
        try:
            keyring.set_password(f"{self.app_name}-{service}", username, password)
            self.logger.info(f"Stored credential for {service}:{username}")
        except Exception as e:
            self.logger.error(f"Failed to store credential: {e}")
            raise
    
    def get_credential(self, service: str, username: str) -> Optional[str]:
        """Retrieve credentials from OS keyring"""
        try:
            return keyring.get_password(f"{self.app_name}-{service}", username)
        except Exception as e:
            self.logger.error(f"Failed to retrieve credential: {e}")
            return None
    
    def delete_credential(self, service: str, username: str):
        """Delete credentials from OS keyring"""
        try:
            keyring.delete_password(f"{self.app_name}-{service}", username)
            self.logger.info(f"Deleted credential for {service}:{username}")
        except Exception as e:
            self.logger.warning(f"Failed to delete credential: {e}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info("Configuration saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            raise
    
    def update_config(self, updates: Dict[str, Any]):
        """Update specific configuration values"""
        config = self.load_config()
        config.update(updates)
        self.save_config(config)
    
    def get_ebay_credentials(self) -> Dict[str, str]:
        """Get eBay API credentials"""
        return {
            "client_id": self.get_credential("ebay", "client_id") or "",
            "client_secret": self.get_credential("ebay", "client_secret") or "",
            "refresh_token": self.get_credential("ebay", "refresh_token") or "",
            "access_token": self.get_credential("ebay", "access_token") or ""
        }
    
    def store_ebay_credentials(self, client_id: str, client_secret: str, 
                              refresh_token: str = "", access_token: str = ""):
        """Store eBay API credentials"""
        self.store_credential("ebay", "client_id", client_id)
        self.store_credential("ebay", "client_secret", client_secret)
        if refresh_token:
            self.store_credential("ebay", "refresh_token", refresh_token)
        if access_token:
            self.store_credential("ebay", "access_token", access_token)
    
    def get_keepa_credentials(self) -> Dict[str, str]:
        """Get Keepa API credentials"""
        return {
            "api_key": self.get_credential("keepa", "api_key") or ""
        }
    
    def store_keepa_credentials(self, api_key: str):
        """Store Keepa API credentials"""
        self.store_credential("keepa", "api_key", api_key)
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get current environment configuration"""
        config = self.load_config()
        return {
            "environment": config.get("environment", "sandbox"),
            "ebay_configured": config.get("ebay_configured", False),
            "keepa_configured": config.get("keepa_configured", False),
            "seller_verified": config.get("seller_verified", False),
            "business_policies_configured": config.get("business_policies_configured", False),
            "opted_in_programs": config.get("opted_in_programs", [])
        }
    
    def set_environment(self, environment: str):
        """Set the environment (sandbox/production)"""
        if environment not in ["sandbox", "production"]:
            raise ValueError("Environment must be 'sandbox' or 'production'")
        
        self.update_config({"environment": environment})
        self.logger.info(f"Environment set to {environment}")
    
    def validate_production_readiness(self) -> List[str]:
        """Validate if configuration is ready for production"""
        issues = []
        
        # Check credentials
        ebay_creds = self.get_ebay_credentials()
        if not ebay_creds["client_id"]:
            issues.append("eBay Client ID not configured")
        if not ebay_creds["client_secret"]:
            issues.append("eBay Client Secret not configured")
        if not ebay_creds["refresh_token"]:
            issues.append("eBay Refresh Token not configured")
        
        # Check environment settings
        env_info = self.get_environment_info()
        if not env_info["seller_verified"]:
            issues.append("eBay seller verification not completed")
        if not env_info["business_policies_configured"]:
            issues.append("eBay business policies not configured")
        
        return issues
    
    def reset_configuration(self):
        """Reset all configuration (for troubleshooting)"""
        try:
            # Remove config file
            if self.config_file.exists():
                self.config_file.unlink()
            
            # Remove setup completed marker
            if self.setup_completed_file.exists():
                self.setup_completed_file.unlink()
            
            # Clear stored credentials
            services = ["ebay", "keepa"]
            usernames = ["client_id", "client_secret", "refresh_token", "access_token", "api_key"]
            
            for service in services:
                for username in usernames:
                    try:
                        self.delete_credential(service, username)
                    except:
                        pass  # Ignore if credential doesn't exist
            
            self.logger.info("Configuration reset completed")
            
        except Exception as e:
            self.logger.error(f"Error during configuration reset: {e}")
            raise
    
    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export configuration for backup or sharing"""
        config = self.load_config()
        export_data = {
            "config": config,
            "environment_info": self.get_environment_info(),
            "setup_completed": not self.is_first_run()
        }
        
        if include_secrets:
            # Only include for backup purposes, not sharing
            export_data["credentials"] = {
                "ebay": self.get_ebay_credentials(),
                "keepa": self.get_keepa_credentials()
            }
        
        return export_data
    
    def import_config(self, import_data: Dict[str, Any], import_secrets: bool = False):
        """Import configuration from backup"""
        try:
            # Import main config
            if "config" in import_data:
                self.save_config(import_data["config"])
            
            # Import credentials if included and requested
            if import_secrets and "credentials" in import_data:
                ebay_creds = import_data["credentials"].get("ebay", {})
                if ebay_creds.get("client_id"):
                    self.store_ebay_credentials(
                        ebay_creds["client_id"],
                        ebay_creds["client_secret"],
                        ebay_creds.get("refresh_token", ""),
                        ebay_creds.get("access_token", "")
                    )
                
                keepa_creds = import_data["credentials"].get("keepa", {})
                if keepa_creds.get("api_key"):
                    self.store_keepa_credentials(keepa_creds["api_key"])
            
            # Mark setup as complete if it was in the import
            if import_data.get("setup_completed", False):
                self.mark_setup_complete()
            
            self.logger.info("Configuration imported successfully")
            
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            raise

# Global configuration instance
production_config = ProductionConfig()
