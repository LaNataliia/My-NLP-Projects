This project is designed to collect the lyrics of an artist and generate a song immitating the artist's style. This is implemented on the basis of the frequency distribution of the words bi-grams in the lyrics.


## Part 1: Getting Lyrics

The code in `get_lyrics.py` gets lyrics of 100 songs of a chosen artist by scraping them from `www.lyrics.com`.
Run the file from the Command Line (Terminal) passing 2 arguments:
- the link to the chosen artist on `www.lyrics.com`,
- the name of the `.txt` file which will be created to save the lyrics.

For example, to get the lyrics of **Queen**:
`..text-generation>python get_lyrics.py https://www.lyrics.com/artist/Queen/5205 queen_lyrics.txt`


## Part 2: Generating a Song

Run `song_writer.py` file from the Command Line (Terminal) passing 2 arguments:
- the name of the `.txt` file created before
- the first word of the future song

For example:
`..text-generation>python song_writer.py queen_lyrics.txt We`
  
  
**Example of the generated text**:  
*We 'll take more - an appetite  
The old fashioned word goes on people people , I 've loved the world is on  
I want to me higher high , we will , we do is about the terror of magic in my face , yeah , yeah  
The old time and me out  
And I 'm gon na be  
The show  
You do  
You got a million women  
And love love give you do n't stop at my life 's late , we will I 'll keep me , we 're a million mirrors  
I 'm havin ' around the dust  
The ogre-men are you to the street  
The light up  
And I want , I 've yet to me higher  
You do  
I 'm here ( people keep calling me higher , yeah yeah , I 'm just do  
I 'll grow  
The people on people on the night  
The ogre-men are telling you  
The old man ask  
Keep yourself he  
You do ba beh beh  
The old man , give you do is ourselves under pressure we will rock you , I have to the terror of my heart ca n't fool I 'm gon na nah*  

