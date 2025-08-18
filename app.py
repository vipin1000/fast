from selenium import webdriver
from selenium.webdriver.common.by import By

# Set up the WebDriver (e.g., Chrome)
driver = webdriver.Chrome()

# Open a webpage
driver.get("https://www.example.com")

# Find an element and interact with it
element = driver.find_element(By.NAME, "q")  # Example: search box
element.send_keys("Selenium Python")
element.submit()

# Close the browser
driver.quit()