from bs4 import BeautifulSoup as bs
import requests
#import pytube
from pytube import YouTube
i = 0
count = 0
base = "https://www.youtube.com/results?search_query="
query = "nissan+advert"
base += query

for i in range(1):
    r = requests.get(base)

    page = r.text
    soup=bs(page,'html.parser')

    vids = soup.findAll('a',attrs={'class':'yt-uix-tile-link'})

    videolist=[]
    for v in vids:
        tmp = 'https://www.youtube.com' + v['href']
        videolist.append(tmp)
    
    print('There are ',len(videolist),' videos returned for page '+str(i+1))

    for item in videolist:
        try:
            # increment counter:
            count+=1
 
            # initiate the class:
            yt = YouTube(item)
        
            #formats = yt.streams.all()
        
            mp4video = yt.streams.filter(progressive=True,file_extension='mp4',resolution='360p')
            #print (mp4video.all())
        
            stream = mp4video.first()
            
            # set the output file name:
            #yt.set_filename(query+' '+str(count))
        
            stream.download('/local/scratch/vanden/Music/Car/')
            
            print('Downloaded video '+str(count))


            # have a look at the different formats available:
            #formats = yt.get_videos()
 
            # grab the video:
            #video = yt.get('mp4', '360p')
 
            # download the video:
            #video.download('./')
        except Exception as e:
            print("Error: ",e,"\n","Count: ",count)
            continue

    # find the navigation buttons in the page html:
    buttons = soup.findAll('a',attrs={'class':"yt-uix-button vve-check yt-uix-sessionlink yt-uix-button-default yt-uix-button-size-default"})
 
    # the button for the next page is the last one in the list:
    nextbutton = buttons[-1]

    # get the url of the next page:
    base = 'https://www.youtube.com' + nextbutton['href']
	  