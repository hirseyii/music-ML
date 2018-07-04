"""
-Naim Sen-
Jun 18
"""
# A script to scrape youtube for videos given a query. Adapted from work
# done previously for A. Clarke.
from bs4 import BeautifulSoup as bs
import requests
import os
import time
import datetime
from pytube import YouTube


"""
A function to grab the upload date of a youtube video from HTML. The age of the
video is calculated (in yrs?) and compared to the max upload age. Returns True
if the upload age is less than or equal to the max upload age, returns false otherwise.
"""
def IsYounger(vid_url, max_upload_age):
    #
    if max_upload_age == None:
        return True
    elif type(max_upload_age) != int or max_upload_age < 1:
        raise ValueError("IsYounger() : Invalid argument - max_upload_age should be a positive integer")
    try:
        watch_page = requests.get(vid_url).text
    except Exception as e:
        print("IsYounger() : request failed ",e)
        exit(1)

    soup = bs(watch_page, 'html.parser')
    date_element = soup.findAll(class_='watch-time-text')
    date_text = date_element[0].text
    # grab year only
    upload_year = int(date_text.split()[-1])
    current_year = datetime.datetime.now().year
    # calculate video age
    upload_age = current_year - upload_year

    # return values
    if upload_age <= max_upload_age:
        return True
    else:
        return False


"""
A function to scrape audio from youtube videos given a set of queries passed as
a string of terms separated by '+' or ' '. Scrapes page by page, (20 videos per
page). Creates new directory in CWD to store audio files. Can filter by video
upload age in years.
"""
def ScrapeAudio(query, num_videos, save_path=None, max_upload_age=None):
    # Check arguments
    if type(num_videos) is not int or num_videos <= 0:
        raise ValueError("ScrapeAudio() : Invalid argument - num_videos should be a positive integer")

    if type(query) is not str:
        raise ValueError("ScrapeAudio() : Invalid argument - query should be a string with terms separated by \'+\'")
    if ' ' in query:
        query = query.replace(' ', '+')

    # save_path is optional and can be auto generated if left blank
    if save_path is None:
            save_path = os.getcwd()+'/SCRAPES_'+query.replace('+', '_')
    # max_upload_age is optional, None=no filter on upload date.
    if max_upload_age == None:
        pass
    elif type(max_upload_age) is not int or max_upload_age < 1:
        raise ValueError("ScrapeAudio() : Invalid argument - max_upload_age should be a positive integer")

    # declare counters
    download_count = 0
    parsed_count = 0
    page_counter = 0

    # base
    base = "https://www.youtube.com/results?search_query="+query

    # For saving to file we need to make a directory if it doesn't exist already
    # and check the file is empty etc.
    # if the path exists check it's empty
    if os.path.isdir(save_path):
        if os.listdir(save_path) == []:
            pass
        else:
            # if directory is not empty throw ENOTEMPTY exception
            raise OSError(39, "Directory is not empty.", save_path)
    else:
        os.makedirs(save_path)

    # get scrape start time
    scrape_start_t = time.time()

    # Loop is broken when download_count = num_videos
    while True:
        # grab page and parse html
        r = requests.get(base)
        page=r.text
        soup = bs(page,'html.parser')
        # grab video links from thumbnail links
        vids = soup.findAll('a', attrs={'class':'yt-uix-tile-link'})
        # create a list of relevant URLS
        videolist = []
        for v in vids:
            # parse href attribute for 'http' regex to skip google adverts
            if (v['href'][0:4] == 'http'):
                continue
            tmp = 'https://www.youtube.com' + v['href']
            videolist.append(tmp)
        print("There are ",len(videolist)," videos returned for page "+str(page_counter+1))
        # loop over video (YT) objects in each page
        for video_url in videolist:
            parsed_count += 1
            try:
                # initialise youtube object
                yt = YouTube(video_url)
                # check video upload age
                if IsYounger(video_url, max_upload_age):
                    # filter AV stream
                    stream = yt.streams.filter(progressive=True,file_extension='mp4',resolution='360p').first()
                    # download audio from stream
                    # check whether title already exists
                    if os.path.isfile(save_path+'/'+stream.default_filename):
                        stream.download(save_path, filename=yt.title+' ('+str(download_count+1)+')')
                    else:
                        stream.download(save_path)
                    # Increment counter
                    download_count += 1
                    print('Downloaded video '+str(download_count))

                    # Check if download_count = num_videos
                    if download_count == num_videos:
                        scrape_end_t = time.time()
                        print("{0} of {1} videos downloaded.\n".format(download_count, parsed_count))
                        print("total time: {0} seconds".format(scrape_end_t-scrape_start_t))
                        return
            except Exception as e:
                print("Error: ",e,"\n","download_count: ",download_count)
                continue

        # Next page
        # find the navigation buttons in the page html:
        buttons = soup.findAll('a',attrs={'class':"yt-uix-button vve-check yt-uix-sessionlink yt-uix-button-default yt-uix-button-size-default"})
        # the button for the next page is the last one in the list:
        nextbutton = buttons[-1]
        # get the url of the next page:
        base = 'https://www.youtube.com' + nextbutton['href']
        page_counter += 1


if __name__ == '__main__':
    ScrapeAudio('strongbow advert', 40)
