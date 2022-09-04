package main

import (
	"log"
	"fmt"
	"net/http"
	"os"
	"github.com/joho/godotenv"
	"github.com/tensorflow/tensorflow/tensorflow/go"
	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"
)

func init() {
    err := godotenv.Load(".env")

    if err != nil {
        log.Fatal("Error loading .env file")
    }
}

func main() {
	bot, err := tgbotapi.NewBotAPI(os.Getenv("TELEGRAM_BOT_API_KEY"))
	if err != nil {
		log.Panic(err)
	}

	bot.Debug = false

	log.Printf("Authorized on account %s", bot.Self.UserName)

	// 
	modelGraph, labels, err := loadModel()
	if err != nil {
		log.Fatalf("unable to load model: %v", err)
	}

	// Create a session for inference over modelGraph
	session, err := tensorflow.NewSession(modelGraph, nil)
	if err != nil {
		log.Fatalf("could not init session: %v", err)
	}

	u := tgbotapi.NewUpdate(0)
	u.Timeout = 60

	updates := bot.GetUpdatesChan(u)

	for update := range updates {		
		if update.Message != nil { // If we got a message
			
			if update.Message.Photo != nil && len(update.Message.Photo) > 0 {
				replyToMessage(bot, update.Message.Chat.ID, "Photo received. Trying to recognize...", -1)
				
				// Get the photo information in max resolution.
				lastPhotoSize := update.Message.Photo[len(update.Message.Photo) - 1]

				// Fetch the photo URL from Telegram.
				fileURL, err := bot.GetFileDirectURL(lastPhotoSize.FileID)
				if err != nil {
					log.Printf("Cannot fetch photo URL: %v", err)
					continue
				}
				
				// Load the photo from URL.
				response, err := http.Get(fileURL)
				if err != nil {
					log.Printf("Cannot load photo from URL: %v", err)
					continue
				}
				defer response.Body.Close()
				
				// Trying to recognize the submitted photo. 
				name, probability := getTensorFlowProbability(response.Body, modelGraph, session, labels)
				
				replyToMessage(bot, update.Message.Chat.ID, fmt.Sprintf("%s (probability: %.2f%%)\n", name, probability), update.Message.MessageID)
			} else {
				// Return an error in the chat if no photo was provided.
				replyToMessage(bot, update.Message.Chat.ID, "Error! You should post an image!", update.Message.MessageID)
			}
		}
	}
}

func replyToMessage(bot *tgbotapi.BotAPI, chatId int64, messageString string, replyMessageId int) {
	message := tgbotapi.NewMessage(chatId, messageString)
	if replyMessageId != -1 {
		message.ReplyToMessageID = replyMessageId
	}
				
	bot.Send(message)
}