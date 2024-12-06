# gp-test

사색 졸업작품

로컬에서 실행하기 위해 파일을 다운받을 때 zip 파일로 한번에 다운받은 다음 AI 모델은 별도로 RAW 파일을 다운받고 zip 파일로 받은 모델 파일은 삭제해야합니다.

<br>

## 텍스트, 이미지 생성 관련

- OpenAI API를 활용하여 텍스트, 이미지 생성 기능을 구현하였습니다.

- Python openai 라이브러리 다운 방법
  ```
  pip install openai
  ```
  ```
  pip install --upgrade openai
  ```

- API 키를 생성하기 위해서는 ChatGPT가 아닌 OpenAI 사이트로 들어가야 합니다.

- 참고한 사이트 - https://platform.openai.com/docs/guides/images/usage, https://github.com/openai/openai-python
 
### api 키 생성하는 방법

- 사이트 접속 - https://platform.openai.com/docs/overview
- 화면 좌측 API keys 클릭
- API 생성


### 보안 관련

- 보안 문제가 있는 내용은 환경 변수로 감춰두었습니다.

- .env 파일을 만들고 그 안에 아래와 같은 코드를 넣으면 됩니다.
  ```
  OPENAI_API_KEY=your_openai_api_key
  SECRET_KEY=your_flask_secret_key
  DB_HOST=your_database_host
  DB_USER=your_database_user
  DB_PASSWORD=your_database_password
  DB_NAME=your_database_name
  ```


<br>

## MySQL DB 생성

- 커넥션 설정은 app.py 코드를 보고 해주시고 테이블 설정은 다음 과정을 따릅니다.
  ```
  CREATE DATABASE user_info;
  ```
  
  ```
  USE user_info;
  ```
  
  ```
  CREATE TABLE users (
      id INT AUTO_INCREMENT PRIMARY KEY,
      user_id VARCHAR(255) UNIQUE NOT NULL,
      password VARCHAR(255) NOT NULL,
      email VARCHAR(255) NOT NULL
  );
  ```
  
  ```
  CREATE TABLE predictions (
      predict_id INT AUTO_INCREMENT PRIMARY KEY,
      user_id VARCHAR(255),
      prediction VARCHAR(50),
      prediction_time DATETIME,
      FOREIGN KEY (user_id) REFERENCES users(user_id)
  );
  ```
  ```
  CREATE TABLE conversations (
    conversation_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(255),
    conversation_name VARCHAR(255),
    created_at DATETIME DEFAULT NOW()
  );
  ```
  ```
  CREATE TABLE messages (
    message_id INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    sender VARCHAR(50), -- 'user' 또는 'assistant'
    message TEXT,
    created_at DATETIME DEFAULT NOW(),
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
  );
  ```
  ```
  ALTER TABLE predictions ADD COLUMN img VARCHAR(255);
  ```
  ```
  ALTER TABLE predictions ADD COLUMN explains TEXT;
  ```
  ```
  ALTER TABLE messages ADD COLUMN raw_markdown TEXT;
  ```














