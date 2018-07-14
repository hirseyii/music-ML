"""
-Naim Sen-
Jun 18
"""
# A script to scrape youtube for videos given a query. Adapted from work
# done previously for A. Clarke.
# This script requires at least PyTube 2.2.3 which is available via the
# PyTube Github repo. This is so that we can make use of the youtube.length
# attribute for filtering by duration.
from bs4 import BeautifulSoup as bs
import requests
import os
import time
import datetime
from pytube import YouTube


# A function to grab the upload date of a youtube video from HTML. The age of the
# video is calculated (in yrs?) and compared to the max upload age. Returns True
# if the upload age is less than or equal to the max upload age, returns false otherwise.

def IsYounger(vid_url, max_upload_age):
    if max_upload_age is None:
        return True
    elif type(max_upload_age) != int or max_upload_age < 1:
        raise ValueError("IsYounger() : Invalid argument - max_upload_age should be a positive integer")
    try:
        watch_page = requests.get(vid_url).text
    except Exception as e:
        print("IsYounger() : request failed ", e)
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


# Returns true if a video's runtime is shorter than the specified max_length (in seconds)
def IsCorrectLength(yt_object, min_length=0, max_length=0):
    # check args
    if min_length > max_length:
        raise ValueError("min_length should be less than max_length dummy!")

    # grab length from the yt object.
    if yt_object.length > max_length:
        return False
    elif max_length is None:
        return True
    else:
        return True


# A function to scrape audio from youtube videos given a set of queries passed as
# a string of terms separated by '+' or ' '. Scrapes page by page, (20 videos per
# page). Creates new directory in CWD to store audio files. Can filter by video
# upload age in years and video length.
# The force_in_title flag can be set to force searching using YouTube's intitle search flag.
def ScrapeVideo(query, num_videos, save_path=None, max_upload_age=None,
                max_length=None, force_in_title=True, check_directory=True):
    # define a few parameters that aren't often tweaked
    min_duration = 30
    max_duration = 6000

    # Check arguments
    if type(num_videos) is not int or num_videos <= 0:
        raise ValueError("ScrapeVideo() : Invalid argument - num_videos should be a positive integer")

    if type(query) is not str:
        raise ValueError("ScrapeVideo() : Invalid argument - query should be a string with terms separated by \'+\'")
    if ' ' in query:
        query = query.replace(' ', '+')

    # save_path is optional and can be auto generated if left blank
    if save_path is None:
            save_path = os.getcwd()+'/SCRAPES_'+query.replace('+', '_')
    # max_upload_age is optional, None=no filter on upload date.
    if max_upload_age is None:
        pass
    elif type(max_upload_age) is not int or max_upload_age < 1:
        raise ValueError("ScrapeVideo() : Invalid argument - max_upload_age should be a positive integer")

    # declare counters
    download_count = 0
    parsed_count = 0
    page_counter = 0

    # switch base depending on force_in_title flag. Makes the first query search with
    # intitle: set
    if force_in_title:
        base = "https://www.youtube.com/results?search_query=intitle%3A"+query
    elif not force_in_title:
        base = "https://www.youtube.com/results?search_query="+query

    # For saving to file we need to make a directory if it doesn't exist already
    # and check the file is empty etc.
    # if the path exists check it's empty
    if os.path.isdir(save_path):
        if os.listdir(save_path) == []:
            pass
        elif check_directory:
            # if directory is not empty ask for confimration
            valid_response = False
            while not valid_response:
                response = input("The directory : {0} is not empty. Are you sure you wish to proceed? Y/N\n".format(save_path))
                if response.lower() == 'y':
                    valid_response = True
                elif response.lower() == 'n':
                    valid_response = True
                    print("The program will now quit.")
                    exit(1)
                else:
                    print("Invalid response, please try again.")
                    valid_response = False

    else:
        os.makedirs(save_path)

    # get scrape start time
    scrape_start_t = time.time()

    # Loop is broken when download_count = num_videos
    while True:
        # grab page and parse html
        r = requests.get(base)
        page = r.text
        soup = bs(page, 'html.parser')
        # grab video links from thumbnail links
        vids = soup.findAll('a', attrs={'class': 'yt-uix-tile-link'})
        # create a list of relevant URLS
        videolist = []
        for v in vids:
            # parse href attribute for 'http' regex to skip google adverts
            if (v['href'][0:4] == 'http'):
                continue
            tmp = 'https://www.youtube.com' + v['href']
            videolist.append(tmp)
        print("There are ", len(videolist), " videos returned for page "+str(page_counter+1))
        # loop over video (YT) objects in each page
        for video_url in videolist:
            parsed_count += 1
            try:
                # initialise youtube object
                yt = YouTube(video_url)
                # check video upload age
                if IsYounger(video_url, max_upload_age) and IsCorrectLength(yt, min_duration, max_duration):
                    # filter AV stream
                    stream = yt.streams.filter(progressive=True, file_extension='mp4', resolution='360p').first()
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
                        print("total time: {0} seconds".format(scrape_end_t - scrape_start_t))
                        return
            except Exception as e:
                print("Error: ", e, "\n", "download_count: ", download_count)
                continue

        # Next page
        # find the navigation buttons in the page html:
        buttons = soup.findAll('a', attrs={'class': "yt-uix-button vve-check yt-uix-sessionlink yt-uix-button-default yt-uix-button-size-default"})
        # the button for the next page is the last one in the list:
        nextbutton = buttons[-1]
        # get the url of the next page:
        base = 'https://www.youtube.com' + nextbutton['href']
        page_counter += 1


if __name__ == '__main__':

    print("pytube version : ", pytube.__version__)
    
    ScrapeVideo('chanel advert', 100, save_path='/raid/scratch/sen/adverts/perfume/', max_upload_age=5)
    ScrapeVideo('dior advert', 100, save_path='/raid/scratch/sen/adverts/perfume/', max_upload_age=5, check_directory=False)
    ScrapeVideo('gucci perfume advert', 100, save_path='/raid/scratch/sen/adverts/perfume/', max_upload_age=5, check_directory=False)
    ScrapeVideo('bvlgari perfume advert', 100, save_path='/raid/scratch/sen/adverts/perfume/', max_upload_age=5, check_directory=False)
    ScrapeVideo('hugo boss perfume advert', 100, save_path='/raid/scratch/sen/adverts/perfume/', max_upload_age=5, check_directory=False)

    ScrapeVideo('strongbow advert', 100, save_path='/raid/scratch/sen/adverts/alcohol/', max_upload_age=5)
    ScrapeVideo('carling advert', 100, save_path='/raid/scratch/sen/adverts/alcohol/', max_upload_age=5, check_directory=False)
    ScrapeVideo('carlsberg advert', 100, save_path='/raid/scratch/sen/adverts/alcohol/', max_upload_age=5, check_directory=False)
    ScrapeVideo('fosters advert', 100, save_path='/raid/scratch/sen/adverts/alcohol/', max_upload_age=5, check_directory=False)
    ScrapeVideo('heineken advert', 100, save_path='/raid/scratch/sen/adverts/alcohol/', max_upload_age=5, check_directory=False)

    ScrapeVideo('nissan advert', 100, save_path='/raid/scratch/sen/adverts/cars/', max_upload_age=5, check_directory=False)
    ScrapeVideo('renault advert', 100, save_path='/raid/scratch/sen/adverts/cars/', max_upload_age=5, check_directory=False)
    ScrapeVideo('audi advert', 100, save_path='/raid/scratch/sen/adverts/cars/', max_upload_age=5, check_directory=False)
    ScrapeVideo('honda advert', 100, save_path='/raid/scratch/sen/adverts/cars/', max_upload_age=5, check_directory=False)
    ScrapeVideo('vw advert', 100, save_path='/raid/scratch/sen/adverts/cars/', max_upload_age=5, check_directory=False)
