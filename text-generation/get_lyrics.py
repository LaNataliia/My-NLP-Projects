"""
Get lyrics of 100 songs of a given artist
"""

import sys
import requests
import lxml
from lxml import html



def getlyrics(artistlink, outfile):
    """Get lyrics for a given artist"""
    
    # get the list of links
    web_page_1 = requests.get(artistlink)
    tree1 = html.fromstring(web_page_1.text)
    all_songlinks = tree1.xpath('//tr/td/strong/a//@href')
    
    #prepare file for saving song lyrics
    with open(outfile, "w") as out_file:
    
        # get the lyrics AND the song name at each link
        for i in range (100):
            songlink = "https://www.lyrics.com/"+str(all_songlinks[i])
            web_page_2 = requests.get(songlink)
            tree2 = html.fromstring(web_page_2.text)
            
            lyrics = tree2.xpath('//*[@id="lyric-body-text"]//text()')
            songname = tree2.xpath('//*[@id="lyric-title-text"]//text()')
            
            formatted_lyrics = ''.join(lyrics)
            formatted_songname = (' '.join(songname)).upper()
            
            #save the lyrics to a file
            out_file.write("SONG " + str(i+1))
            out_file.write("\n")
            out_file.write(formatted_songname)
            out_file.write("\n\n\n")
            out_file.write(formatted_lyrics)
            out_file.write("\n\n\n\n\n")
            

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: " + sys.argv[0] + " <link to the artist page on www.lyrics.com> <outfile name>")
    else:
        getlyrics(sys.argv[1], sys.argv[2])


            
        


