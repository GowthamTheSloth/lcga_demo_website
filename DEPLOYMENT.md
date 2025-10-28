# 🚀 Deploy Your LCGA Demo to a Permanent Website

Choose one of these free hosting options to make your website permanently accessible:

## Option 1: Railway (Recommended - Easiest)

### Steps:
1. **Go to [Railway.app](https://railway.app)** and sign up with GitHub
2. **Click "New Project"** → "Deploy from GitHub repo"
3. **Connect your GitHub account** and select this repository
4. **Railway will automatically detect** the Python app and deploy it
5. **Your website will be live** at a URL like: `https://your-app-name.railway.app`

### What happens:
- Railway automatically installs dependencies from `requirements.txt`
- Uses the `Procfile` to start your app
- Provides a permanent HTTPS URL
- Free tier includes 500 hours/month

---

## Option 2: Render (Also Great)

### Steps:
1. **Go to [Render.com](https://render.com)** and sign up
2. **Click "New +"** → "Web Service"
3. **Connect your GitHub** and select this repository
4. **Configure:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
5. **Click "Create Web Service"**
6. **Your website will be live** at: `https://your-app-name.onrender.com`

### What happens:
- Render uses `render.yaml` for configuration
- Free tier includes 750 hours/month
- Automatic HTTPS

---

## Option 3: Heroku (Classic)

### Steps:
1. **Install Heroku CLI** from [heroku.com](https://devcenter.heroku.com/articles/heroku-cli)
2. **Login to Heroku:**
   ```bash
   heroku login
   ```
3. **Create a new app:**
   ```bash
   heroku create your-lcga-demo-name
   ```
4. **Deploy:**
   ```bash
   git add .
   git commit -m "Deploy LCGA demo"
   git push heroku main
   ```
5. **Open your app:**
   ```bash
   heroku open
   ```

### What happens:
- Uses `Procfile` and `runtime.txt`
- Free tier available (with limitations)
- URL: `https://your-app-name.herokuapp.com`

---

## Option 4: PythonAnywhere (Simple)

### Steps:
1. **Go to [PythonAnywhere.com](https://www.pythonanywhere.com)** and sign up
2. **Upload your files** via the Files tab
3. **Create a new Web App** (Flask)
4. **Set the source code directory** to your project folder
5. **Configure WSGI file** to point to your `app.py`
6. **Reload the web app**

---

## 🎯 Recommended: Railway

**Why Railway is best for you:**
- ✅ **Easiest setup** - just connect GitHub
- ✅ **Automatic deployment** - no manual commands
- ✅ **Free tier** - 500 hours/month
- ✅ **Custom domain** - can add your own domain later
- ✅ **Automatic HTTPS** - secure by default
- ✅ **No credit card required**

## 📋 Before Deploying

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git push -u origin main
   ```

2. **Make sure all files are included:**
   - ✅ `app.py`
   - ✅ `requirements.txt`
   - ✅ `Procfile`
   - ✅ `Dockerfile`
   - ✅ `railway.json`
   - ✅ All folders: `algorithms/`, `templates/`, `static/`, `utils/`

## 🔧 After Deployment

Your website will be permanently accessible at a URL like:
- Railway: `https://lcga-demo.railway.app`
- Render: `https://lcga-demo.onrender.com`
- Heroku: `https://lcga-demo.herokuapp.com`

## 🎉 Benefits of Permanent Hosting

- **No more localhost** - accessible from anywhere
- **Share with others** - send them the URL
- **Always online** - runs 24/7
- **Professional** - looks like a real website
- **HTTPS secure** - safe for users

## 🆘 Troubleshooting

### Common Issues:
1. **Build fails**: Check `requirements.txt` has all dependencies
2. **App crashes**: Check logs in your hosting platform's dashboard
3. **Slow loading**: Reduce algorithm parameters for faster execution
4. **Memory issues**: Lower population size and generations

### Need Help?
- Railway: Check their [docs](https://docs.railway.app)
- Render: Check their [docs](https://render.com/docs)
- Heroku: Check their [docs](https://devcenter.heroku.com)

---

**Ready to deploy? Choose Railway for the easiest experience!** 🚀


