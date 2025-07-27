# vit



```
$ssh-keygen -t rsa -b 4096 -C "getsnigeria@gmail.com"
```


Assuming you are connecting GitHub over SSH, you can run below command to confirm this.

```
$git config --get remote.origin.url
```

If you get a result has following format git@github.com:xxx/xxx.github.com.git, then you should do the following.

Generate a SSH key(or use existing one). if you had one, you just
need to add your key to the ssh-agent (step 2)and to your GitHub
account(step 3).

below are for those who don't have SSH key.

**Step 1** Generating public/private rsa key pair.

```
$ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

You'll be asked to confirm where to save the SSH key and what passphrase you want to use.

**Step 2** Add your key to the ssh-agent

* Ensure ssh-agent is enabled
  `$eval "$(ssh-agent -s)"`
* Add your SSH key to the ssh-agent:
  `$ssh-add ~/.ssh/id_rsa`

**Step 3** Add your SSH key to your account

`$sudo apt-get install xclip`

`$xclip -sel clip < ~/.ssh/id_rsa.pub`

Then add the copied key to GitHub

Go to  **Settings** -> **SSH keys** (Personal settings side bar)-> **Add SSH key** ->fill out form(key is on your clipboard, just use ctrl+v)->**Add key**
