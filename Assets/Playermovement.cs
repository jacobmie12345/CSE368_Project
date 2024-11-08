using UnityEngine;

public class Playermovement : MonoBehaviour
{
    private new Camera camera;
	private new Rigidbody2D rigidbody;
	
	private Vector2 velocity;
	private float inputAxis;
	
	public float moveSpeed = 8f;
	
	
	private void Awake()
	{
		rigidbody = GetComponent<Rigidbody2D>();
		camera = Camera.main;
	}
	
	private void Update()
	{
		HorizontalMovement();
	}
	
	private void HorizontalMovement()
	{
		inputAxis = Input.GetAxis("Horizontal");
		velocity.x = Mathf.MoveTowards(velocity.x, inputAxis * moveSpeed, moveSpeed * Time.deltaTime);
	}
	
	private void FixedUpdate()
	{
		Vector2 position = rigidbody.position;
		position += velocity * Time.fixedDeltaTime;
		
		Vector2 leftEdge = camera.ScreenToWorldPoint(Vector2.zero);
		Vector2 rightEdge = camera.ScreenToWorldPoint(new Vector2(Screen.width, Screen.height));
		position.x = Mathf.Clamp(position.x, leftEdge.x, rightEdge.x);
		
		rigidbody.MovePosition(position);
	}
}
