
set DIR="D:/Shuo/affdex/new-hci"
echo DIR=%DIR%

for /R %DIR% %%f in (*.avi) do (
	D:\Shuo\ffmpeg\bin\ffmpeg -i %%f %%f.mp4
)