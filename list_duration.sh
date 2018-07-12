find "$1" -type f | while read videoPath ; do
    videoFile=$(basename "$videoPath")
    duration=$(ffmpeg -i "$videoPath" 2>&1 | grep Duration)

    echo -e "$videoFile:\n $duration"
done
