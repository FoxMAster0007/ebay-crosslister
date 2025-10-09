# eBay CrossLister - API Configuration Guide

**Complete setup guide for enterprise-grade Amazon to eBay automation**

This guide walks you through setting up the required API integrations for professional operation. All credentials are stored securely in your system's keyring - never in plain text files.

---

## Overview

**Required for basic operation:**
- **eBay Developer API** - Creates listings automatically  
- **Keepa API** - Cross-references scraped Amazon data (optional but strongly suggested)

---

## 1. eBay Developer API Setup

### **Step 1: Create Developer Account**
1. Visit: **https://developer.ebay.com**
2. Click "Create Account" or sign in with existing eBay account
3. Complete developer registration form
4. Verify your email address

### **Step 2: Create Application**
1. Go to "My Account" → "Applications"
2. Click "Create Application"
3. **Application Details:**
   - **App Name**: "eBay CrossLister Professional"
   - **App Type**: "Server-to-Server"
   - **Application Purpose**: "Product listing automation"
   - **APIs Needed**: Select "Inventory API" and "Account API"

### **Step 3: Get Credentials**
Once approved (usually instant):
1. Go to "Application Keys"
2. **Copy these values:**
   - **Client ID** (App ID) - Your public identifier
   - **Client Secret** - Your private key (keep secure!)

### **Step 4: Environment Selection**
- **Sandbox** (Recommended for testing):
  - Safe testing environment
  - No real listings created
  - Perfect for initial setup
- **Production** (For real business):
  - Creates actual eBay listings
  - Requires OAuth flow for user consent
  - Use after testing in sandbox

---

## 🔍 2. Keepa API Setup (Highly Recommended)

### **Why Keepa?**
- **99.9% reliability** vs 60-70% web scraping success
- **No IP blocking** - Direct API access
- **Historical price data** - Better pricing decisions
- **Fast response times** - 200ms vs 2-5s scraping
- **Rich product data** - Reviews, categories, variations

### **Setup Process:**
1. **Visit**: https://keepa.com
2. **Create Account**: Free registration
3. **Purchase API Tokens**:
   - Navigate to "Data Access" → "API"
   - **Starter Package**: 100,000 tokens (~€50)
   - **Cost per request**: ~€0.0005 (very affordable)
   - **ROI**: Pays for itself with first few sales

4. **Get API Key**:
   - Dashboard → "API" section
   - Copy your API key (starts with numbers)

### **Token Usage:**
- **Product lookup**: 1 token per ASIN
- **Historical data**: 1 additional token
- **Price tracking**: 1 token per check
- **Typical usage**: 50-100 tokens per listing created

---

## 3. Professional Pricing Configuration

### **Profit Margin Settings**

**Net Profit Margin** (excludes eBay and PayPal fees):
 - Set this to what you'd like your additional profit margin to be

**eBay Fee Rate**:
- the incorporated rate adjustment for eBay
- **Current Standard**: 0.125 (12.5%) for most categories
- **Electronics**: 0.08-0.10 (8-10%) 
- **Fashion**: 0.12-0.15 (12-15%)
- **Books/Media**: 0.15 (15%)
- **Update regularly**: eBay changes fees periodically

**PayPal/Payment Processing**:
- the incorporated rate adjustment for PayPal
- **PayPal Standard**: 0.0349 (3.49%) + $0.49 per transaction
- **eBay Managed Payments**: 0.03 (3.0%) average
- **Stripe/Others**: 0.029 (2.9%) + $0.30

### **Advanced Pricing Strategies**

**Dynamic Pricing Factors**:
- **Competition Level**: Adjust margins based on market saturation
- **Product Velocity**: Lower margins for fast-moving items
- **Seasonal Demand**: Higher margins during peak seasons
- **Brand Premium**: Luxury brands can command higher margins

---

## 🔧 4. Application Configuration

### **Initial Setup Process**

1. **Launch Application**:
   ```bash
   ./run_crosslister.sh (or ./run_crosslister.bat) # If running from launch script
   # OR
   python main.py        # If running from source
   ```

2. **Open Configuration** (First run will prompt automatically):
   - Press `c` for configuration menu
   - If you haven't already, go through the Setup Wizard and configure your eBay account (accessible via "Launch Setup Wizard" in the Accounts tab, check README.md for more info)
   - After eBay developer and seller accounts configuration - scroll down in the Accounts tab to the Keepa section to enter your Keepa API Key:
      ```
      Keepa API Key: [paste-your-keepa-key-here]
      ```

3. **Configure Pricing**:
   Within the Pricing tab
   ```
   Net Profit Margin: 0.25 (25%)
   eBay Fee Rate: 0.125 (12.5%)
   Payment Fee Rate: 0.0349 (3.49%)
   ```

4. **Test Configuration**:
   - Import test ASIN: `B0F3PT1VBL` (popular electronics)
   - Verify data loads correctly
   - Check calculated pricing makes sense
   - Test eBay connection (sandbox first)

### **Environment Switching**

**Sandbox Mode** (Safe Testing):
- No real listings created
- Safe for experimentation  
- Full API testing capability
- No financial risk

**Production Mode** (Live Business):
- Creates real eBay listings
- Charges eBay fees
- Requires careful testing first
- Full business operation

## Security Notes

- All credentials stored in OS keyring (Windows Credential Manager, macOS Keychain, Linux Secret Service)
- Never stored in plain text files
- App uses HTTPS for all API calls
- Rate limiting protects your API quotas

## Troubleshooting

### Keepa Issues
- **Invalid API key**: Double-check key from keepa.com dashboard
- **Rate limit exceeded**: Wait 1 minute between requests
- **No credits**: Purchase more API tokens

### eBay Issues  
- **Authentication failed**: Refresh OAuth tokens
- **Listing rejected**: Check eBay category policies
- **Duplicate listing**: Product might already be listed

### General Issues
- **Network errors**: Check internet connection
- **App crashes**: Restart and check logs
- **Config not saving**: Check OS keyring permissions

## Getting Help

1. Check the app logs (visible in the app)
2. Verify API credentials in respective dashboards
3. Test APIs independently first
4. Submit a support request with specific error messages - send to contact@sylcrala.xyz

## Quick Test Sequence

1. Configure Keepa API key
2. Import test ASIN: B0F3PT1VBL (Sony headphones)
3. Click "Optimize All"
4. Verify calculated price makes sense
5. Configure eBay credentials
6. Test listing (start with test item)

## Pro Tips

- Start with small Keepa token package - if opting to use Keepa
- Test with low-value items first  
- Monitor eBay fees - they change occasionally
- Keep profit margins realistic (15-25%)
- Use different categories to test fee rates
