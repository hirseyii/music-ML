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
    if type(max_upload_age) != int or max_upload_age < 1:
        raise ValueError('CompareUploadDate() : Invalid argument - max_upload_age should be a positive integer')
    try:
        watch_page = requests.get(vid_url).text
    except Exception as e:
        print("CompareUploadDate() : request failed ",e)
        exit(1)

    soup = bs(watch_page, 'html.parser')
    date_element = soup.findAll(class_="watch-time-text")
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
    if type(num_videos) is not int or num_pages <= 0:
        raise ValueError('ScrapeAudio() : Invalid argument - num_videos should be a positive integer')

    if type(query) is not str:
        raise ValueError('ScrapeAudio() : Invalid argument - query should be a string with terms separated by \'+\'')
    if ' ' in query:
        query = query.replace(' ', '+')

    # save_path is optional and can be auto generated if left blank
    if save_path is None:
            save_path = os.getcwd()+'/SCRAPES_'+query.replace('+', '_')
    # max_upload_age is optional, None=no filter on upload date.
    if max_upload_age is not int or max_upload_age < 1:
        raise ValueError('ScrapeAudio() : Invalid argument - max_video_age should be a positive integer')

    # declare counters

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


if __name__ == '__main__':
    print(IsYounger("https://www.youtube.com/watch?v=s03I6DEjgbc", 5))
