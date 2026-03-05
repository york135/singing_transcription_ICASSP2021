import yt_dlp
import os
import sys
import json

if __name__ == '__main__':

    yt_link_path = sys.argv[1]
    train_set_dir = sys.argv[2]
    test_set_dir = sys.argv[3]

    if not os.path.isdir(train_set_dir):
        os.mkdir(train_set_dir)

    if not os.path.isdir(test_set_dir):
        os.mkdir(test_set_dir)

    with open(yt_link_path, newline='') as file:
        yt_link = json.load(file)

    # For easier download tracking
    success = 0
    failed = 0

    # Songs #1~#400 are training set 
    for i in range(1, 401):

        out_dir = os.path.join(train_set_dir, str(i))

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        outtmpl = os.path.join(out_dir, "Mixture")

        if os.path.isfile(outtmpl):
            print("Song id", i, "already downloaded, skipping")
            success += 1
            continue

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "320"
            }],
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([yt_link[str(i)]])
                print("Song id", i, "OK")
                success += 1
        except Exception as e:
            print("Song id", i, "not available -", str(e)[:60])
            failed += 1
            # Remove the created empty directory if download failed
            if os.path.isdir(out_dir) and not os.listdir(out_dir):
                os.rmdir(out_dir)

    # Songs #401~#500 are test set 
    for i in range(401, 501):

        out_dir = os.path.join(test_set_dir, str(i))

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        outtmpl = os.path.join(out_dir, "Mixture")

        if os.path.isfile(outtmpl):
            print("Song id", i, "already downloaded, skipping")
            success += 1
            continue

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "320"
            }],
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([yt_link[str(i)]])
                print("Song id", i, "OK")
                success += 1
        except Exception as e:
            print("Song id", i, "not available -", str(e)[:60])
            failed += 1
            if os.path.isdir(out_dir) and not os.listdir(out_dir):
                os.rmdir(out_dir)

    print(f"\nDone! {success} downloaded, {failed} failed")