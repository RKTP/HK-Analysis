import pymysql
import sys
import pickle as pkl
import os
import os.path as path
import time
from datetime import datetime

data_dir = '/home/hkcrawl/data'
base_article_path = os.path.join(data_dir,"articles")
base_meta_path = os.path.join(data_dir,"meta")
batch_size = 300

with open("article_keyword_matrix.pkl", "rb") as f:
    aids = pkl.loads(f.read())

with open("keyword_index.pkl", "rb") as f:
    keywords = pkl.loads(f.read())

class DBHandlerFactory():
    def __init__(self, host, port, user, pwd, db):
        self.host = host
        self.port = port
        self.user = user
        self.password = pwd
        self.db = db

    def create_handler(self):
        conn = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, db=self.db, charset='utf8', local_infile=True)
        return SqlHandler(conn)


class SqlHandler():
    def __init__(self, conn):
        self.conn = conn

    def commit(self):
        self.conn.commit()
        return

    def rollback(self):
        self.conn.rollback()
        return


    def upload_articles(self, article_metainfo):
        c = self.conn.cursor()
        query = "INSERT IGNORE INTO `Article` (aid, title, timestamp, contentPath, score, url, uploadedTimestamp) VALUES (%s, %s, now()+0, %s, %s, %s, %s)"
        start_index = 0
        while start_index < len(article_metainfo):
            for aid in list(article_metainfo.keys())[start_index:start_index+batch_size]:
                uptimestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(article_metainfo[aid][4]))
                params = [aid, article_metainfo[aid][0], article_metainfo[aid][1], article_metainfo[aid][2], article_metainfo[aid][3], uptimestamp]
                c.execute(query, params)

            self.commit()
            start_index += batch_size

        c.close()

    def upload_article_features(self, features):
        c = self.conn.cursor()
        query = "INSERT INTO `Feature` (Aid, InvertedIndex) VALUES (%s, %s) ON DUPLICATE KEY UPDATE InvertedIndex=%s"
        for aid in list(features.keys()):
            params = [aid, pkl.dumps(features[aid]), pkl.dumps(features[aid])]
            c.execute(query, params)
            self.commit()

        c.close()

    def upload_keywords(self, in_keywords):
        c = self.conn.cursor()
        query = "INSERT IGNORE INTO `Keyword` (Kid, Keyword) VALUES (%s, %s)"
        start_index = 0
        while start_index < len(in_keywords):
            for kid in list(keywords.keys())[start_index:start_index+batch_size]:
                params = [kid, in_keywords[kid]]
                c.execute(query, params)
            self.commit()
            start_index += batch_size

        c.close()


input_meta = {}
for aid in list(aids.keys()):
    with open(path.join(base_meta_path, aid+".pkl"), "rb") as f:
        metadata = pkl.loads(f.read())
        input_meta[aid] = [metadata['title'], path.join(base_article_path, aid+".pkl"), metadata['score'], metadata['url'], metadata['time']]


dbf = DBHandlerFactory("news-recommendation.cvynbgusgzyb.ap-northeast-2.rds.amazonaws.com", 3306, "master", "weonforall", "innodb").create_handler()

dbf.upload_articles(input_meta)
dbf.upload_article_features(aids)
dbf.upload_keywords(keywords)
dbf.conn.close()
