# eBay CrossLister
## Version: 1.0

**Automate your eBay listings**  
*Eliminating subscriptions with privacy in mind.*

Want to list Amazon products on your own eBay store? Don't worry, with this local script you can just input a corresponding ASIN and let us do the rest! We'll pull all data from the listing per ASIN, correlate and confirm the data in each listing, and push it to a linked eBay seller account as individual listings!


## Quick Start

### Simple Installation (Recommended)
```bash
# Download repository
git clone https://github.com/sylcrala/ebay_crosslister.git
cd ebay_crosslister

# Windows installation
install.bat

# Unix / Linux / macOS installation
./install.sh
```

### Development Setup
```bash
# For developers and customization
# Install with developer-specific tools

# Windows
install.bat --dev

# Unix / Linux / macOS
./install.sh --dev    

# All platforms (direct python launch)
python main.py              
```

### Running the Application
After installation, use any of these methods:
```bash
# Using generated shortcuts (created during install)
./run_crosslister.sh        # Unix / macOS / Linux
run_crosslister.bat         # Windows (double-click or command line)

# Direct Python execution
python main.py

# With command line options
python main.py --config     # Open configuration screen
python main.py --setup      # Run setup wizard  
python main.py --reset      # Reset all configuration
python main.py --help       # Show all options
```

### Python-only install (if preferred)
```bash
# If you prefer to use the Python installer directly
python3 install.py          # Standard installation
python3 install.py --dev    # With development tools
python3 install.py --force  # Force clean reinstall
```

### Manual Install (if needed)
```bash
# Manual setup for advanced users
python3 -m venv .venv
source .venv/bin/activate     # Unix / Linux / macOS 
# OR .venv\Scripts\activate   # Windows
pip install -r requirements.txt
python main.py
```


**Features:**
- **Unlimited listings** - List as many as you'd like, keeping your PC specs in mind!
- **Local data control** - Never send data to third parties
- **Enterprise security** - All credentials stored in your OS keychain
- **Full source access** - Customize for your business needs


## Core Features

### **Automated Product Pipeline**
- **Bulk Amazon Import**: ASINs, URLs, search results - hundreds at once
- **Smart Data Enhancement**: Auto-optimized titles, descriptions, categories  
- **Pricing Optimization**: Optimizes all products based on designated fees and profit margins
- **One-Click eBay Listing**: Automated category selection & batch processing
- **Real-Time Analytics**: Track margins, success rates, performance

### **Enterprise Security**
- **Credential Protection**: OS keyring integration (never plain text)
- **Zero Data Leakage**: Everything stays local, no cloud dependencies
- **OAuth Standards**: Professional API authentication
- **Input Sanitization**: Comprehensive validation prevents attacks
- **Smart Error Recovery**: Automatic retry logic with exponential backoff

### **Professional Advantages**
- **Cross-Platform Native**: Installs and runs smoothly on any platform - Includes .bat files for easy Windows operation
- **Modern TUI Interface**: Fast, responsive, professional appearance
- **Scalable Architecture**: Handle thousands of products efficiently
- **Full Customization**: Source code access for business-specific needs

### **Production-Tested Reliability**
- **Production-Ready**: Comprehensive testing across all major platforms
- **High Performance**: Optimized for speed and memory efficiency
- **Professional Logging**: Detailed audit trails and error reporting
- **Self-Diagnostics**: Automatic recovery from API failures and network issues

## Configuration

### eBay API Setup

#### Step 1.
1. Get eBay Developer Account at [developer.ebay.com](https://developer.ebay.com)
2. Create application and get Client ID/Secret
3. Use "Advanced Setup" in application settings
4. Enter your custom credentials
   - **Account API** (recommended)
5. Submit application and wait for approval

#### Step 2: Get API Credentials
Once approved:
1. Create a new application with your dev account, name it something along the lines of "crosslister".
2. Go to "Application Keysets"
3. **Important**: Choose your keyset environment:
   - **Sandbox**: For testing (default, recommended initially - ensure that you've selected "Sandbox Mode" within the application if you're opting for sandbox testing - otherwise continue to production environment)
   - **Production**: For real listings (required for application functionality)
4. Copy your **Client ID** (App ID) for the newly made application
5. Copy your **Client Secret** for the newly made application
6. Go to "User Access Tokens"
7. Remember which keyset environment you're setting up, select accordingly under the dropdown menu.
8. Once on the User Access Tokens screen for your application under the appropriate environment, scroll down to "Get a Token from eBay via your Application" and set up a new link.
9. After completed, copy the **ruName** shown for the application configuration
10. Lastly, opt out of eBay marketplace account notifications as described within the link below - our application does not save user data:
   - https://developer.ebay.com/marketplace-account-deletion#optingOut

#### Step 3. Set up business policies
After you've set up your eBay Developer account and gathered all required information, the last thing we need to do prior to configuring the application is set up some business policies on your eBay seller account - as our app uses these for creating the listings. While the main business policy required on our end is the shipping policy (as we pull this information on-demand per listing, in order to determine which shipping method is appropriate based on product weight and value) - it's also important to have the return and payment policies configured in this section for listing creation process.

Here is a link to the eBay Business Policies description page, which will describe the process of creating and managing your policies - Make sure you opt in to the Seller Hub in order to start using business policies!
 - https://www.ebay.com/help/selling/business-policies/business-policies?id=4212 

#### Step 4: Configure in App
1. Run the app: `python main.py`
2. Press `c` for configuration
3. Go through the setup wizard - it should launch upon first start:
   - If the wizard doesnt auto-launch, it should be accessible under the "accounts" tab within the config
   - Follow the guide contained within the wizard, it will ask for the API keys and ruName that we extracted during the previous step.

#### Step 5: Testing vs Production

**Sandbox Mode (Recommended for Testing):**
- Creates fake listings on sandbox.ebay.com
- No real money or inventory involved
- Perfect for testing the application
- Uses Client Credentials authentication

**Production Mode (Real Listings):**
- Creates real listings on ebay.com
- Requires user consent via OAuth flow
- You'll need to implement OAuth to get refresh tokens
- **Note**: Production mode requires additional setup beyond this app

##### Sandbox Testing Limitations
- Sandbox listings are not visible on real eBay
- Some features may behave differently than production
- Perfect for validating the integration works

##### Optional: Keepa Account  
1. Sign up at [Keepa.com](https://keepa.com)
2. Get your API key from account settings
3. Provides more accurate Amazon data than web scraping


## Usage

### Basic Workflow
1. **Import Products**: Enter Amazon ASINs 
2. **Review Data**: Check product information and pricing
3. **Optimize Prices**: Automatic pricing based on eBay fees
4. **List to eBay**: One-click listing creation

### Price Optimization
- Default: 20% markup over Amazon price
- Automatically accounts for eBay fees (~12-15%)
- Customizable profit margins
- Competitive pricing analysis (with Keepa)


## Security

- API keys stored in system keyring (not plain text)
- Input validation prevents injection attacks
- Rate limiting protects API quotas  
- Secure HTTPS connections only
- No data stored on external servers


## Troubleshooting

### Common Issues

**"No Keepa API key configured"**
- Add your Keepa API key in the Settings tab
- Or continue without Keepa (will use web scraping)

**"eBay authentication failed"**  
- Verify your Client ID and Client Secret
- Check eBay Developer console for application status

**"Rate limit exceeded"**
- Wait a few minutes between large imports
- Keepa has usage limits on free accounts

### Logs
Check the application logs in the "Logs" tab for detailed error information.


## Troubleshooting

### eBay API Issues

**"Failed to get access token"**
- Check your Client ID and Client Secret are correct
- Ensure you're using the right environment (sandbox vs production)
- Verify your eBay developer account is approved

**"Failed to create inventory item"**
- Check product has valid title and price
- Ensure you have proper API permissions
- Try with a simpler product first

**"Failed to publish offer"**
- Business policies might be required (in eBay's seller account)
- Check category ID is valid for your product
- Ensure all required fields are populated

**Rate Limiting**
- The app includes built-in delays between API calls
- eBay sandbox allows fewer requests than production
- If you hit limits, wait and try again


### General Issues

**Products not importing**
- Try individual ASINs first to isolate the issue
- Check your internet connection
- Amazon may be blocking requests (try later)

**Price optimization errors**
- Ensure Amazon price was detected correctly
- Check your margin settings aren't too high
- Verify eBay/PayPal fee rates are reasonable


### Getting Help

1. Check the application logs in the bottom panel
2. Try sandbox mode first before production
3. Test with simple, common products initially
4. Join the discussion on GitHub Issues

## Project Structure

```
ebay_crosslister/
├── main.py           # Main application file
├── requirements.txt  # Dependencies
├── setup.py         # Installation configuration  
├── README.md        # This file
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

I apologize for any issues you may find during your usage of this application, I've been learning with the help of Copilot (hence why you can see some "vibe-code" throughout the scripts) but it's a process! If you need any help, want to report any issues, or just make a suggestion - please submit either an Issue or Pull Request on the GitHub repo! 

---

p.s. pls don't roast me for the lack of modularization, I promise I'm modularizing the script properly in a future update! I'm learning!!

**made with love by sylcrala :)**
