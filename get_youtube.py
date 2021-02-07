import youtube_dl
import os
import sys
import json

if __name__ == '__main__':

    yt_id = []
    data_id = []
    offset_list = []

    yt_link_path = sys.argv[1]
    train_set_dir = sys.argv[2]
    test_set_dir = sys.argv[3]

    if not os.path.isdir(train_set_dir):
        os.mkdir(train_set_dir)

    if not os.path.isdir(test_set_dir):
        os.mkdir(test_set_dir)

    with open(yt_link_path, newline='') as file:
        yt_link = json.load(file)

    # song #1~#400 are training set 
    for i in range(1, 401):

        out_dir = os.path.join(train_set_dir, str(i))

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        outtmpl = os.path.join(out_dir, "Mixture.mp3")

        if os.path.isfile(outtmpl):
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
        }

        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                youtube_dl.utils.std_headers['User-Agent'] = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
                ydl.download([yt_link[str(i)]])
                print ("Song id", i, "OK")
        except:
            print ("Song id", i, "not available")
            os.rmdir(out_dir)


    # song #401~#500 are test set 
    for i in range(401, 501):

        out_dir = os.path.join(test_set_dir, str(i))

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        outtmpl = os.path.join(out_dir, "Mixture.mp3")

        if os.path.isfile(outtmpl):
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
        }

        try:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                youtube_dl.utils.std_headers['User-Agent'] = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
                ydl.download([yt_link[str(i)]])
                print ("Song id", i, "OK")
        except:
            print ("Song id", i, "not available")