/**
 * Instagram Post Screenshot Generator
 * 
 * Generates 1080x1350px screenshots from HTML files for Instagram posts.
 * 
 * INSTALLATION:
 * -------------
 * 1. Install Node.js from https://nodejs.org/ (if not already installed)
 * 
 * 2. Install Puppeteer:
 *    npm install puppeteer
 * 
 * USAGE:
 * ------
 * node export.js
 * 
 * OUTPUT:
 * -------
 * Creates 'instagram_post.png' in the same directory (1080x1350px)
 * 
 * REQUIREMENTS:
 * -------------
 * - Node.js v14 or higher
 * - puppeteer v21.0.0 or higher
 * 
 * TROUBLESHOOTING:
 * ----------------
 * If you get "Cannot find module 'puppeteer'":
 *   npm install puppeteer
 * 
 * If screenshot is wrong size:
 *   Check deviceScaleFactor is set to 1
 */

const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch({
    defaultViewport: null
  });
  
  const page = await browser.newPage();
  
  // Set exact viewport with 1x device scale
  await page.setViewport({
    width: 1080,
    height: 1350,
    deviceScaleFactor: 1  // Force 1x resolution (not 2x Retina)
  });
  
  // Load HTML file - UPDATE THIS PATH to your HTML file
  await page.goto('file:///Users/galazar/Desktop/website/assets/instagram_exporter/ig_code_post_template.html', {
    waitUntil: 'networkidle0'  // Wait for all network requests to finish
  });
  
  // Wait for fonts to load (2 seconds)
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Save screenshot to current directory
  const outputPath = path.join(__dirname, 'instagram_post.png');
  await page.screenshot({
    path: outputPath,
    clip: {
      x: 0,
      y: 0,
      width: 1080,
      height: 1350
    }
  });
  
  await browser.close();
  console.log(`✅ Screenshot saved to: ${outputPath}`);
})();