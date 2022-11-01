# import asyncio
# import json

# from binance import AsyncClient, DepthCacheManager, BinanceSocketManager, OptionsDepthCacheManager

# async def main():
#     client = await AsyncClient.create()
#     print(json.dumps(await client.get_exchange_info(), indent=2))
#     bsm = BinanceSocketManager(client)
    
#     async with bsm.trade_socket('ETHBTC') as ts:
#         for _ in range(5):
#             res = await ts.recv()
#             print(f'recv {res}')

#     klines = client.get_historical_klines("BNBBTC", AsyncClient.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")

#     async for kline in await client.get_historical_klines_generator("BNBBTC", AsyncClient.KLINE_INTERVAL_1MINUTE, "1 day ago UTC"):
#         print(kline)

#     klines = client.get_historical_klines("ETHBTC", client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")

    
#     klines = client.get_historical_klines("NEOBTC", client.KLINE_INTERVAL_1WEEK, "1 Jan, 2017")

#     async with DepthCacheManager(client, symbol='ETHBTC') as dcm_socket:
#         for _ in range(5):
#             depth_cache = await dcm_socket.recv()
#             print(f"symbol {depth_cache.symbol} updated:{depth_cache.update_time}")
#             print("Top 5 asks:")
#             print(depth_cache.get_asks()[:5])
#             print("Top 5 bids:")
#             print(depth_cache.get_bids()[:5])

#     # Vanilla options Depth Cache works the same, update the symbol to a current one
#     options_symbol = 'BTC-210430-36000-C'
#     async with OptionsDepthCacheManager(client, symbol=options_symbol) as dcm_socket:
#         for _ in range(5):
#             depth_cache = await dcm_socket.recv()
#             count += 1
#             print(f"symbol {depth_cache.symbol} updated:{depth_cache.update_time}")
#             print("Top 5 asks:")
#             print(depth_cache.get_asks()[:5])
#             print("Top 5 bids:")
#             print(depth_cache.get_bids()[:5])

#     await client.close_connection()

# if __name__ == "__main__":

#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(main())
# class test:
#     def testing(self):
#         print("testing...")