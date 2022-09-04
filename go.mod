module github.com/alextataurov/ai-image-recognition-telegrambot

go 1.18

require github.com/tensorflow/tensorflow v2.9.0+incompatible

require (
	github.com/go-telegram-bot-api/telegram-bot-api/v5 v5.5.1 // indirect
	github.com/golang/protobuf v1.5.0 // indirect
	github.com/joho/godotenv v1.4.0 // indirect
	google.golang.org/protobuf v1.28.1 // indirect
)

replace github.com/tensorflow/tensorflow => /home/alextataurov/go/src/github.com/tensorflow/tensorflow
