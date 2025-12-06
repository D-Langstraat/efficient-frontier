How to Run efficient_frontier.py

Pull or download the 2 files:

•	efficient_frontier.py – the Python script (the code)

•	prices.csv – the price data (do not rename or edit this file)

This guide shows you exactly how to run the script on Windows.

________________________________________
1. Install Python (one-time setup)

1.	Open your web browser.
2.	Go to https://www.python.org.
3.	Click Downloads → Download Python 3.x (any recent 3.x version is fine).

Step 2 – Run the installer
When the installer opens:
1.	At the bottom of the first screen, check this box:
✅ “Add Python 3.x to PATH”
2.	Click “Install Now”.
3.	When it finishes, click Close.

Step 3 – Make sure Python is working
1.	Press Start, type PowerShell, and open Windows PowerShell.
2.	Type:
   
3.	python --version
   
You should see something like:
Python 3.12.1
5.	Then type:

6.	pip --version
7.	
You should see a pip version.
If either one says “not recognized”, reinstall Python and make sure “Add Python to PATH” was checked.

________________________________________
2. Install the required Python packages
The script uses four add-on libraries:
•	numpy
•	pandas
•	matplotlib
•	scipy
Install them once by running this in PowerShell:

pip install numpy pandas matplotlib scipy

Wait until it finishes (you should see messages like “Successfully installed …”).

________________________________________
3. Set up the project folder on your Desktop
We’re going to put everything in a folder right on your Desktop.
Step 1 – Create the folder
1.	Go to your Desktop.
2.	Right-click → New → Folder.
3.	Name it exactly:
4.	Efficient Frontier

Now you should have:
C:\Users\<YourName>\Desktop\Efficient Frontier
(Windows replaces <YourName> with your actual username.)
Step 2 – Put the two files into that folder
Copy the two files you were sent into this folder:

•	efficient_frontier.py
•	prices.csv

In the end, your Desktop folder should look like:

C:\Users\<YourName>\Desktop\Efficient Frontier\efficient_frontier.py
C:\Users\<YourName>\Desktop\Efficient Frontier\prices.csv
Do not rename either file.

________________________________________
4. Run the script

Step 1 – Open PowerShell in that folder
1.	Open Windows PowerShell.
2.	Type this command (copy/paste is fine), replacing <YourName> with your actual Windows username:
   
3.	cd "C:\Users\<YourName>\Desktop\Efficient Frontier"

Example:
cd "C:\Users\Alex\Desktop\Efficient Frontier"
5.	Press Enter.
Now PowerShell is “inside” the project folder.

Step 2 – Run the Python script
In the same PowerShell window, type:

python "efficient_frontier.py"

Then press Enter.

________________________________________
5. What you should see
If everything is set up correctly:
1.	Two charts will pop up (one after the other):
o	Efficient Frontier (Without Shorting)
o	Efficient Frontier (With Shorting)

2.	Each chart shows:
o	A cloud of random portfolios.
o	The “Markowitz bullet” frontier:
	Bottom part (inefficient portfolios) in gray dashed.
	Top part (efficient portfolios) in orange.
o	A black star at the max Sharpe ratio portfolio.
o	A text box on the right that lists:
	Annual return.
	Annual standard deviation.
	Annual Sharpe ratio.
	Exact weights for each stock in that max-Sharpe portfolio.

3.	Two image files will also be saved automatically to your Desktop:
o	efficient_frontier_without_shorting.png
o	efficient_frontier_with_shorting.png




________________________________________
Quick troubleshooting
A) 'python' is not recognized...
Cause: Python is not on the system PATH.
Fix: Re-run the Python installer and make sure “Add Python 3.x to PATH” is checked. Then test again:
python --version
________________________________________
B) ModuleNotFoundError: No module named 'pandas' (or numpy/matplotlib/scipy)
Cause: The libraries aren’t installed.
Fix: Run:
pip install numpy pandas matplotlib scipy
________________________________________
C) FileNotFoundError: [Errno 2] No such file or directory: 'prices.csv'
Cause: PowerShell is not in the same folder as the files.
Fix:
1.	Make sure both files are here:
2.	C:\Users\<YourName>\Desktop\Efficient Frontier\efficient_frontier.py
3.	C:\Users\<YourName>\Desktop\Efficient Frontier\prices.csv
4.	Make sure you cd into that exact folder before running:
5.	cd "C:\Users\<YourName>\Desktop\Efficient Frontier"
6.	python "efficient_frontier.py"
