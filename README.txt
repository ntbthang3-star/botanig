BOTANIQ - STREAMLIT + RANDOM FOREST CUA NHOM

1. Giao dien
- Botaniq_v20.html: giao dien HTML/CSS/JavaScript goc.
- botaniq_component/frontend/: cau phan Streamlit Component nho de truyen du lieu giua HTML va Python.

2. Thuat toan
- thuat_toan.py: ma MyRandomForest va MyDecisionTree tu cai dat cua nhom, duoc giu nguyen.
- mo_hinh_web.py: doc train.csv, huan luyen MyRandomForest tren cac cot duoc chon, du doan va tinh accuracy tren test.csv.
- Mac dinh moi mo hinh su dung 50 cay, Gini, bootstrap, max_features=sqrt, soft voting, random_state=42.
- Khi nguoi dung chon tap chi so khac nhau, he thong huan luyen mo hinh tuong ung tren dung tap dac trung do.
- Ket qua phan tram la phan bo soft-voting cua cac cay; khong phai xac suat thanh cong ngoai dong ruong.
- Accuracy hien thi duoc tinh tren test.csv cho dung tap dac trung dang chon.

3. Chay tren Windows PowerShell
  py -m venv .venv
  .\.venv\Scripts\Activate.ps1
  py -m pip install -r requirements.txt
  py -m streamlit run app.py

5. Chay kiem thu thuat toan doc lap
  py kiem_thu_mo_hinh.py

4. Thu tai khoan demo tren giao dien goc
- Nong dan: demo@datcay.vn / 123456
- Quan tri: admin@datcay.vn / admin123

Luu y: day la tai khoan demo luu tren localStorage cua trinh duyet, khong phai he thong xac thuc bao mat cho san pham that.
