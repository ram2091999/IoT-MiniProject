from mirai_core import Bot, Updater
from mirai_core.models import Event, Message, Types

qq = 123456
host = '192.168.1.112'
port = 115200
auth_key = 'abcdefgh'
scheme = 'http'  # http or https

bot = Bot(qq, host, port, auth_key, scheme=scheme)
updater = Updater(bot)





@updater.add_handler([Event.Message])
async def handler(event: Event.BaseEvent):
    if isinstance(event, Event.Message):  
        await bot.send_message(target=event.sender,
                               message_type=event.type,
                               message=event.messageChain,
                               quote_source=event.messageChain.get_source())


        message_chain = [

            Message.Plain(text='test')
        ]
        if event.type == Types.MessageType.GROUP:
            message_chain.append(event.member.id)
        image = Message.Image(path='/root/whatever.jpg')
        message_chain.append(image)
        
        bot_message = await bot.send_message(target=event.sender,
                                             message_type=event.type,
                                             message=message_chain,
                                             quote_source=event.messageChain.get_source())


        print(bot_message.messageId)


        image_id = image.imageId
        print(image_id)
        return True

updater.run()