# Accessing a Remote Jupyter Notebook with PuTTY, Pageant, and SSH on Windows

This guide walks you through accessing a Jupyter Notebook running on a remote Linux cluster (e.g., `monteverde.cl.cam.ac.uk`) from a **Windows machine**, using:

- PuTTY (SSH client)
- PuTTYgen (SSH key generator)
- Pageant (SSH agent for key authentication)
- SSH port forwarding

---

## Requirements

- Your remote **username** and **host** (e.g., `username@monteverde.cl.cam.ac.uk`)
- Access to the remote server (or help from an admin to upload your public key)

---

## Step-by-Step Instructions

### Step 1: Install PuTTY Tools

Download from the website:  
[https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)

Install the following:

- PuTTY
- PuTTYgen
- Pageant

Alternatively, use the **MSI installer** to get all three.

---

### Step 2: Generate SSH Key with PuTTYgen

1. Open **PuTTYgen**.
2. Click **Generate** and move your mouse in the box.
3. Click **File-Save private key** (e.g., `key.ppk`).
4. Copy the public key from the top box (starts with `ssh- ...`).

---

### Step 3: Add Public Key to Remote Server

> Send it off to Mark

---

### Step 4: Load Private Key into Pageant

1. Start **Pageant**.
2. In the system tray, right-click Pageant > **Add Key**.
3. Select your `.ppk` file (e.g., `key.ppk`).

Your key is now active for authentication.

---

### Step 5: Configure PuTTY to Connect and Forward Jupyter Port

1. Open **PuTTY**.

2. In the **Session** panel:

   * **Host Name**: `username@monteverde.cl.cam.ac.uk`
   * **Port**: `22`
   * **Connection type**: SSH

3. Go to **Connection > SSH > Auth**:

   * Check **"Allow agent forwarding"**
   * **Don't** set a private key here (Pageant handles it)

4. Go to **Connection > SSH > Tunnels**:

   * **Source port**: `8001`
   * **Destination**: `127.0.0.1:8000`
   * Click **Add**

5. Click **Open** to connect.

### Step 6: Open Jupyter in Your Local Browser

Visit:

```
http://localhost:8001
```
