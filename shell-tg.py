# pytelegrambotapi
import telebot
import basic

bot = telebot.TeleBot('5504387349:AAGVOTGoaSvZ1y6BBRFC7aHi3tMNsC98vco')

# img = open('data/plane.jpg', 'rb')

# Функция, обрабатывающая команду /start
@bot.message_handler(commands=["start"])
def start(m, res=False):
    bot.send_message(m.chat.id, 'Я на связи. Напиши мне что-нибудь )')
    # bot.send_photo(m.chat.id, img);

# Получение сообщений от юзера
@bot.message_handler(content_types=["text"])
def handle_text(message):
    result, err = basic.run('<tg>', message.text)
    if err:
        print()
        bot.send_message(message.chat.id, err.as_string())
    else:
        bot.send_message(message.chat.id, str(result))



# Запускаем бота
bot.polling(none_stop=True, interval=0)